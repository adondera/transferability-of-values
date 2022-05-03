from collections import defaultdict

import numpy as np
import pandas as pd
import os
import tensorflow as tf
import time

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM, GlobalMaxPool1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

from nlp.transferability.utils import combine_datasets, set_seeds, print_results

# some configuration
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
BATCH_SIZE = 128
EPOCHS = 10
THRESHOLD = 0.5
N_SPLITS = 10

moral_labels = ['care', 'harm',
                'fairness', 'cheating',
                'loyalty', 'betrayal',
                'authority', 'subversion',
                'purity', 'degradation',
                'non-moral']

moral_foundations = ['care', 'fairness', 'loyalty', 'authority', 'purity', 'non-moral']

os.environ['TF_DETERMINISTIC_OPS'] = '1'


def evaluate_LSTM(target_domain, do_kfold=True, use_foundations=False, dataset_size=35000, fine_tune=False,
                  train_all=False, target_frac=1, target_experiment=False):
    seed_val = 42
    set_seeds(seed_val)
    tf.random.set_seed(seed_val)
    EMBEDDING = 'glove'
    EMBEDDING_DIM = 300
    EMBEDDING_FILE = 'nlp/data/embeddings/glove.6b.300d.txt'

    print(f'Loading {EMBEDDING} word vectors...')
    word2vec = {}
    with open(EMBEDDING_FILE, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vec
    print('Found %s word vectors.' % len(word2vec))

    source, target = combine_datasets(target_domain, target_frac)
    source_text = source['text']
    source_labels = source[moral_labels].values
    target_text = target['text']
    target_labels = target[moral_labels].values

    sentences = pd.concat((source_text, target_text)).values

    # convert the sentences (strings) into integers
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(sentences)

    source_sequences = tokenizer.texts_to_sequences(source_text.values)
    target_sequences = tokenizer.texts_to_sequences(target_text.values)

    # get word -> integer mapping
    word2idx = tokenizer.word_index
    print('Found %s unique tokens.' % len(word2idx))

    # pad sequences so that we get a N x T matrix
    source_data = pad_sequences(source_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    target_data = pad_sequences(target_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of source data tensor:', source_data.shape)
    print('Shape of target data tensor:', target_data.shape)

    # prepare embedding matrix
    print('Filling pre-trained embeddings...')
    num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word2idx.items():
        if i < MAX_VOCAB_SIZE:
            embedding_vector = word2vec.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all zeros.
                embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(
        num_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False
    )

    target_folds = KFold(10).split(target_text, target_labels)
    # cross validation
    source_folds = KFold(10)

    f1_scores_source = defaultdict(list)
    f1_scores_target = defaultdict(list)
    f1_scores_source_cf = defaultdict(list)

    durations = []

    for train, test in source_folds.split(source_data, source_labels):
        source_X_train, source_X_test = source_data[train], source_data[test]
        source_y_train, source_y_test = source_labels[train], source_labels[test]

        target_train_index, target_test_index = next(target_folds)

        target_X_train, target_X_test = target_data[target_train_index], target_data[target_test_index]
        target_y_train, target_y_test = target_labels[target_train_index], target_labels[target_test_index]

        if train_all:
            source_X_train = np.concatenate((source_X_train, target_X_train))
            source_y_train = np.concatenate((source_y_train, target_y_train))

        print('Building model...')
        # create an LSTM network with a single LSTM
        input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
        x = embedding_layer(input_)
        x = LSTM(15, return_sequences=True)(x)
        x = GlobalMaxPool1D()(x)
        output = Dense(len(moral_labels), activation="sigmoid")(x)

        model = Model(input_, output)
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(lr=0.01),
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        )
        print('Training model...')
        if not target_experiment:
            start_time = time.time()
            r = model.fit(
                source_X_train,
                source_y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                verbose=0
            )
            end_time = time.time()

        print('======== Statistics on test set ========')
        source_y_pred = model.predict(source_X_test)

        source_y_pred[source_y_pred < THRESHOLD] = 0
        source_y_pred[source_y_pred > THRESHOLD] = 1

        source_y_pred = source_y_pred.astype(source_y_test.dtype)

        print(classification_report(source_y_test, source_y_pred, target_names=moral_labels))
        clf_report_source = classification_report(source_y_test, source_y_pred, target_names=moral_labels,
                                                  output_dict=True)

        if fine_tune or target_experiment:
            start_time = time.time()
            r = model.fit(
                target_X_train,
                target_y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                verbose=0
            )
            end_time = time.time()

        print('======== Statistics on target set ========')
        target_y_pred = model.predict(target_X_test)
        target_y_pred[target_y_pred < THRESHOLD] = 0
        target_y_pred[target_y_pred > THRESHOLD] = 1

        print(classification_report(target_y_test, target_y_pred, target_names=moral_labels))
        clf_report_target = classification_report(target_y_test, target_y_pred, target_names=moral_labels,
                                                  output_dict=True)

        print('======== Statistics for catastrophic forgetting ========')
        source_y_pred = model.predict(source_X_test)

        source_y_pred[source_y_pred < THRESHOLD] = 0
        source_y_pred[source_y_pred > THRESHOLD] = 1

        print(classification_report(source_y_test, source_y_pred, target_names=moral_labels))
        clf_report_source_cf = classification_report(source_y_test, source_y_pred, target_names=moral_labels,
                                                     output_dict=True)

        print(f'Time spent training: {end_time - start_time}')
        durations.append(end_time - start_time)

        for label in moral_labels + ['micro avg', 'macro avg', 'weighted avg']:
            if label in clf_report_source:
                f1_scores_source[label].append(clf_report_source[label]['f1-score'])
                f1_scores_target[label].append(clf_report_target[label]['f1-score'])
                f1_scores_source_cf[label].append(clf_report_source_cf[label]['f1-score'])

    print_results(f1_scores_source, f1_scores_target, f1_scores_source_cf, moral_labels)
    print(f'Average time spent training {sum(durations) / len(durations)}')

    return
