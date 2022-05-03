import time
from collections import defaultdict

import fastText as fasttext
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer

from nlp.transferability.utils import set_seeds, combine_datasets, print_results

moral_labels = ['care', 'harm',
                'fairness', 'cheating',
                'loyalty', 'betrayal',
                'authority', 'subversion',
                'purity', 'degradation',
                'non-moral']

moral_foundations = ['care', 'fairness', 'loyalty', 'authority', 'purity', 'non-moral']


# Write file in fasttext format for model input
def write_formatted_file(path_to_file, data, labels):
    f = open(path_to_file, 'w', encoding='utf-8')
    to_write = ''
    formatted_labels = []
    for i in range(len(data)):
        new_line = ''
        new_list = []
        for label in labels[i]:
            new_line += f'__label__{label} '
            new_list.append(f'__label__{label}')
        new_line += data[i].replace('\n', ' ') + '\n'
        to_write += new_line
        formatted_labels.append(new_list)
    f.write(to_write)
    f.close()
    return formatted_labels


def evaluate_fasttext(target_domain, do_kfold=True, use_foundations=False, dataset_size=35000, fine_tune=False,
                      train_all=False, target_frac=1, target_experiment=False):
    seed_val = 42
    set_seeds(seed_val)
    source_data, target_data = combine_datasets(target_domain, target_frac)

    source_text = source_data['text'].to_numpy()[:dataset_size]
    target_text = target_data['text'].to_numpy()[:dataset_size]

    source_labels = np.array(
        [[source_data.columns[i] for i, y in enumerate(x) if y == 1] for x in source_data.to_numpy()])
    target_labels = np.array(
        [[target_data.columns[i] for i, y in enumerate(x) if y == 1] for x in target_data.to_numpy()])

    f1_scores_source = defaultdict(list)
    f1_scores_target = defaultdict(list)
    f1_scores_source_cf = defaultdict(list)

    target_folds = KFold(10).split(target_text, target_labels)
    source_kf = KFold(n_splits=10)
    durations = []
    for train_index, test_index in source_kf.split(source_text, source_labels):
        X_train_source, X_test_source = source_text[train_index], source_text[test_index]
        y_train_source, y_test_source = source_labels[train_index], source_labels[test_index]

        target_train_index, target_test_index = next(target_folds)
        X_train_target, X_test_target = target_text[target_train_index], target_text[target_test_index]
        y_train_target, y_test_target = target_labels[target_train_index], target_labels[target_test_index]

        if train_all:
            X_train_source = np.concatenate((X_train_source, X_train_target))
            y_train_source = np.concatenate((y_train_source, y_train_target))

        write_formatted_file(f'nlp/data/fasttext_input/source.train', X_train_source, y_train_source)
        write_formatted_file(f'nlp/data/fasttext_input/target.train', X_train_target, y_train_target)
        formatted_source_labels = write_formatted_file(f'nlp/data/fasttext_input/source.test', X_test_source,
                                                       y_test_source)
        formatted_target_labels = write_formatted_file(f'nlp/data/fasttext_input/target.test', X_test_target,
                                                       y_test_target)

        start_time = time.time()
        if not target_experiment:
            model = fasttext.train_supervised(input=f"nlp/data/fasttext_input/source.train", lr=0.03, epoch=50,
                                              verbose=0)
        else:
            model = fasttext.train_supervised(input=f"nlp/data/fasttext_input/target.train", lr=0.03, epoch=50,
                                              verbose=0)
        end_time = time.time()

        pred_labels, probabilities = model.predict(X_test_source.tolist(), k=11, threshold=0.3)

        mlb = MultiLabelBinarizer(np.array([f'__label__{x}' for x in moral_labels]))

        mlb.fit(formatted_source_labels + pred_labels)

        target_classes = [c_name[len('__label__'):] for c_name in mlb.classes_]
        print('======== Statistics on test set ========')
        print(classification_report(mlb.transform(formatted_source_labels), mlb.transform(pred_labels),
                                    target_names=target_classes))
        clf_report_source = classification_report(mlb.transform(formatted_source_labels), mlb.transform(pred_labels),
                                                  target_names=target_classes, output_dict=True)

        if fine_tune:
            model.save_model('model.bin')
            start_time = time.time()
            model = fasttext.train_supervised(input=f"nlp/data/fasttext_input/target.train", lr=0.03, epoch=50,
                                              inputModel='model.bin', incr=True, verbose=0)
            end_time = time.time()

        print('======== Statistics on target set ========')
        pred_labels, probabilities = model.predict(X_test_target.tolist(), k=11, threshold=0.3)
        print(classification_report(mlb.transform(formatted_target_labels), mlb.transform(pred_labels),
                                    target_names=target_classes))
        clf_report_target = classification_report(mlb.transform(formatted_target_labels), mlb.transform(pred_labels),
                                                  target_names=target_classes, output_dict=True)

        print('======== Statistics for catastrophic forgetting ========')
        pred_labels, probabilities = model.predict(X_test_source.tolist(), k=11, threshold=0.3)
        print(classification_report(mlb.transform(formatted_source_labels), mlb.transform(pred_labels),
                                    target_names=target_classes))
        clf_report_source_cf = classification_report(mlb.transform(formatted_source_labels), mlb.transform(pred_labels),
                                                     target_names=target_classes, output_dict=True)
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
