import datetime
import time
import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from transformers import AdamW, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from scipy.special import expit
from decimal import Decimal
from nlp.bert.utils import format_time, classification

moral_labels = ['care', 'harm',
                'fairness', 'cheating',
                'loyalty', 'betrayal',
                'authority', 'subversion',
                'purity', 'degradation',
                'non-moral']

DEFAULT_CONFIG = {
    "label_names": moral_labels,
    "epochs": 3,
    "loss_fct": BCEWithLogitsLoss(),
    "threshold": 0,
    "batch_size": 16,
    "optim": AdamW,
    "name": 'bert-base-uncased',
    "dropout": 0.1
}

timestamp = datetime.datetime.now()
timestamp_str = timestamp.strftime("%d-%b(%H-%M)")


class MultiLabelBertBase:
    def __init__(self, config=DEFAULT_CONFIG, model=None):
        self.config = config
        if model:
            self.model = model
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.config['name'],
                                                                            num_labels=len(self.config["label_names"]),
                                                                            hidden_dropout_prob=self.config["dropout"])

        self.optim = self.config['optim'](self.model.parameters(), lr=5e-5)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print(f'We will use the GPU: {torch.cuda.get_device_name(0)}')
        else:
            self.device = torch.device("cpu")
            print('No GPU available, using the CPU instead.')

        self.model.to(self.device)

    def train(self, train_dataset, dev_dataset, validation=True):
        # Move datasets to GPU
        train_dataset.to(self.device)
        dev_dataset.to(self.device)

        self.model.train()
        print("\nTraining...")

        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=self.config['batch_size'])

        validation_step = math.floor(len(train_loader) / 4)

        best_loss = 100

        epoch_t0 = time.time()

        total_steps = len(train_loader) * self.config['epochs']

        scheduler = get_linear_schedule_with_warmup(self.optim, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(self.config['epochs']):
            print(f'\n======== Epoch {epoch + 1} / {self.config["epochs"]} ========\n')

            batch_count = 0
            for batch in train_loader:
                self.optim.zero_grad()
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels'].float()
                output = self.model(input_ids, attention_mask=attention_mask)
                loss = self.config["loss_fct"](output.logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optim.step()
                scheduler.step()

                batch_count += 1

                # Validation step
                if batch_count % validation_step == 0 and validation:
                    elapsed = format_time(time.time() - epoch_t0)
                    y_predicted, y_true, loss = self.evaluate(dev_loader)
                    if loss < best_loss:
                        best_loss = loss
                        predictions = (y_predicted > self.config['threshold']).float().cpu().numpy()
                        true_labels = y_true.cpu().numpy()
                        print('\nClassification Report:\n')
                        print(classification(true_labels, predictions, target_names=self.config['label_names']))

                    print(f'Batch {batch_count}  of  {len(train_loader)}    '
                          f'Average loss: {round(Decimal(loss), 3)}    '
                          f'Elapsed {elapsed}')

        print(f'Total training time: {format_time(time.time() - epoch_t0)}')

    def evaluate(self, loader):
        self.model.eval()
        y_predicted = []
        y_true = []
        total_loss = 0
        for batch in loader:
            with torch.no_grad():
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels'].float()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = self.config["loss_fct"](outputs.logits, labels)
                total_loss += loss.item()
            logits = outputs.logits
            true_values = batch['labels']
            y_predicted.extend(logits)
            y_true.extend(true_values)
        self.model.train()
        return torch.stack(y_predicted), torch.stack(y_true), total_loss / len(loader)

    def predict(self, dataset, save=False):
        dataset.to(self.device)
        loader = DataLoader(dataset, batch_size=self.config['batch_size'])
        y_predicted, y_true, loss = self.evaluate(loader)
        print(f'Prediction loss: {loss}')
        if save:
            np.savetxt('predicted.txt', expit(y_predicted.cpu().numpy()), fmt='%1.2f')
            np.savetxt('true.txt', y_true.cpu().numpy(), fmt='%1d')
        predictions = (y_predicted > self.config['threshold']).float().cpu().numpy()
        return predictions, y_true.cpu().numpy()

    def load_config(self, new_config):
        self.config.update(new_config)
