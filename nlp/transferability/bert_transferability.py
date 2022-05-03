import json
import random
from collections import defaultdict
import time
import numpy as np

import torch

from nlp.bert.data import BertDataset
from nlp.bert.model import MultiLabelBertBase
from nlp.transferability.utils import set_seeds, combine_datasets, classification, print_results, get_dataset

from torch.nn import BCEWithLogitsLoss
from transformers import AutoTokenizer
from transformers import AdamW

moral_labels = ['care', 'harm',
                'fairness', 'cheating',
                'loyalty', 'betrayal',
                'authority', 'subversion',
                'purity', 'degradation',
                'non-moral']

moral_foundations = ['care', 'fairness', 'loyalty', 'authority', 'purity', 'non-moral']


def evaluate_bert(target_domain, source_domains=[], do_kfold=True, use_foundations=False, dataset_size=35000, fine_tune=False,
                  train_all=False, target_frac=1, target_experiment=False):
    model_name = 'bert-base-uncased'

    # Model config
    MODEL_CONFIG = {
        "label_names": moral_foundations if use_foundations else moral_labels,
        "epochs": 3,
        "loss_fct": BCEWithLogitsLoss(),
        "threshold": 0,
        "batch_size": 16,
        "optim": AdamW,
        "name": model_name,
        "dropout": 0.1
    }

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    set_seeds(seed_val)

    if len(source_domains) > 0:
        print(f"Evaluating with source domains: {source_domains}")
        source_data = get_dataset(source_domains)
        target_data = get_dataset([target_domain])
    else:
        source_data, target_data = combine_datasets(target_domain)

    BertDataset.tokenizer = AutoTokenizer.from_pretrained(model_name)

    source_dataset = BertDataset(data_file=source_data, use_foundations=use_foundations,
                                 label_names=MODEL_CONFIG.get('label_names'), max_size=dataset_size)
    target_dataset = BertDataset(data_file=target_data, use_foundations=use_foundations,
                                 label_names=MODEL_CONFIG.get('label_names'), max_size=dataset_size)

    if use_foundations:
        print('Using the foundations labels')
    else:
        print('Using the moral values labels')

    if do_kfold:
        f1_scores_source = defaultdict(list)
        f1_scores_target = defaultdict(list)
        f1_scores_source_cf = defaultdict(list)

        all_source_classifications = []
        all_target_classifications = []

        target_folds = iter(target_dataset.kfold(10))
        durations = []
        for train_dataset, test_dataset in source_dataset.kfold(10):

            target_train_dataset, target_test_dataset = next(target_folds)
            # number_of_samples = int(target_frac * len(target_dataset))
            # shuffled_indices = random.sample(range(len(target_train_dataset.text)), number_of_samples)
            # new_texts = [target_train_dataset.text[i] for i in shuffled_indices]
            # new_labels = target_train_dataset.labels[shuffled_indices]
            # target_train_dataset = BertDataset(data_file=None, texts=new_texts, labels=new_labels)

            if train_all:
                print('Adding the target domain train set to the whole training set...')
                train_dataset = BertDataset(data_file=None,
                                            texts=train_dataset.text + target_train_dataset.text,
                                            labels=torch.cat((train_dataset.labels, target_train_dataset.labels)),
                                            ids=np.concatenate((train_dataset.ids, target_train_dataset.ids)))

            bert = MultiLabelBertBase(config=MODEL_CONFIG)
            if not target_experiment:
                start_time = time.time()
                bert.train(train_dataset, test_dataset, validation=False)
                end_time = time.time()
            print('======== Statistics on test set ========')
            clf_report_source, _ = classification(bert, test_dataset, labels=MODEL_CONFIG['label_names'])
            print('======== Statistics on target set ========')
            if fine_tune or target_experiment:
                print('Fine tuning model on out of domain dataset...')
                start_time = time.time()
                bert.train(target_train_dataset, target_test_dataset, validation=False)
                end_time = time.time()
            clf_report_target, target_classifications = classification(bert, target_test_dataset,
                                                                       labels=MODEL_CONFIG['label_names'])
            print('======== Statistics for catastrophic forgetting ========')
            clf_report_source_cf, source_classifications = classification(bert, test_dataset,
                                                                          labels=MODEL_CONFIG['label_names'])
            print(f'Time spent training: {end_time - start_time}')
            durations.append(end_time - start_time)

            all_target_classifications.extend(target_classifications)
            all_source_classifications.extend(source_classifications)

            for label in MODEL_CONFIG["label_names"] + ['micro avg', 'macro avg', 'weighted avg']:
                if label in clf_report_target:
                    f1_scores_source[label].append(clf_report_source[label]['f1-score'])
                    f1_scores_target[label].append(clf_report_target[label]['f1-score'])
                    f1_scores_source_cf[label].append(clf_report_source_cf[label]['f1-score'])

        print_results(f1_scores_source, f1_scores_target, f1_scores_source_cf, MODEL_CONFIG['label_names'])
        print(f'Average time spent training {sum(durations) / len(durations)}')

        method = 'source'
        if fine_tune:
            method = 'finetune'
        elif target_experiment:
            method = 'target'
        elif train_all:
            method = 'trainall'

        # Write all predictions to json
        with open(f'bert_{target_domain}_{method}_source.json', 'w') as file:
            json.dump(all_source_classifications, file)
        with open(f'bert_{target_domain}_{method}_target.json', 'w') as file:
            json.dump(all_target_classifications, file)

        return
