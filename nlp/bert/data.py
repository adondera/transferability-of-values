import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split, KFold

# The moral foundations dict with virtue as key and vice as value
MORAL_FOUNDATIONS = {
    'care': 'harm',
    'fairness': 'cheating',
    'loyalty': 'betrayal',
    'authority': 'subversion',
    'purity': 'degradation',
    'non-moral': 'non-moral'
}


class MFTCDataset(Dataset):
    def __init__(self, data_file=None, use_foundations=False, label_names=None, texts=None, labels=None, ids=None,
                 max_size=None):
        self.ids = ids
        self.text = texts
        self.labels = labels
        self.label_names = label_names

        if data_file is not None:
            if isinstance(data_file, str):
                self.data = pd.read_csv(data_file).dropna()
            else:
                self.data = data_file

            if max_size is None:
                max_size = self.data.shape[0]

            self.text = self.data['text'].to_list()[:max_size]
            self.ids = self.data['tweet_id'].to_numpy()[:max_size]

            if use_foundations:
                if self.label_names is None:
                    self.label_names = MORAL_FOUNDATIONS.keys()

                for l_name in self.label_names:
                    if l_name not in MORAL_FOUNDATIONS:
                        raise KeyError(f'Foundation {l_name} does not exist')

                num_rows = min(self.data.shape[0], max_size)
                num_cols = len(self.label_names)
                self.labels = np.empty([num_rows, num_cols], dtype=bool)
                for i, key in enumerate(self.label_names):
                    value = MORAL_FOUNDATIONS.get(key)
                    self.labels[:, i] = self.data[key].to_numpy()[:max_size] | self.data[value].to_numpy()[:max_size]
                self.labels = self.labels.astype(int)
            else:
                if self.label_names is None:
                    self.label_names = [x for x in self.data.columns if x != 'text']

                for l_name in self.label_names:
                    if l_name not in self.data.columns:
                        raise KeyError(f'Moral label {l_name} does not exist')

                self.labels = self.data[self.label_names].to_numpy()[:max_size]

    def __getitem__(self, index):
        return {'id': self.ids[index],
                'text': self.text[index],
                'labels': self.labels[index]}

    def __len__(self):
        return len(self.data)


class BertDataset(MFTCDataset):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def __init__(self, data_file=None, use_foundations=False, label_names=None, texts=None, labels=None, ids=None,
                 max_size=35000):
        super().__init__(data_file, use_foundations, label_names, texts, labels, ids, max_size)

        self.encodings = BertDataset.tokenizer(self.text, truncation=True, padding=True, max_length=64,
                                               return_tensors='pt')
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        item['id'] = self.ids[idx]
        return item

    def __len__(self):
        return len(self.labels)

    def kfold(self, k=10):
        kf = KFold(n_splits=k)

        texts = np.array(self.text)
        labels = self.labels
        ids = self.ids

        for train_index, test_index in kf.split(X=self.text, y=self.labels):
            X_train, X_test = texts[train_index], texts[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            ids_train, ids_test = ids[train_index], ids[test_index]

            train_dataset = BertDataset(data_file=None, texts=X_train.tolist(), labels=y_train, ids=ids_train,
                                        label_names=self.label_names)
            test_dataset = BertDataset(data_file=None, texts=X_test.tolist(), labels=y_test, ids=ids_test,
                                       label_names=self.label_names)

            yield train_dataset, test_dataset

    def to(self, device):
        self.encodings = self.encodings.to(device)
        self.labels = self.labels.to(device)

    @staticmethod
    def get_tokenization(string):
        return BertDataset.tokenizer(string, return_tensors='pt')
