import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split

DATASETS = {'diabetes': {'target': 'diabetes_mellitus', 'criterion': nn.CrossEntropyLoss(), 'num_classes': 2},
            'airbnb': {'target': 'price', 'criterion': nn.MSELoss(), 'num_classes': 1},
            'har': {'target': 'Activity', 'criterion': nn.CrossEntropyLoss(), 'num_classes': 6},
            'compas': {'target': 'two_year_recid', 'criterion': nn.CrossEntropyLoss(), 'num_classes': 2},
            'MNIST': {'target': 'class_label', 'criterion': nn.CrossEntropyLoss(), 'num_classes': 10},
            'CelebA': {'target': 'gender', 'criterion': nn.CrossEntropyLoss(), 'num_classes': 2}}

class CustomDataset(Dataset):
    def __init__(self, data, target, features=None, concepts=None):
        self.target = target

        if features is None:
            self.features = list(data.columns.difference([target]))
        else:
            self.features = features
        
        self.X = torch.tensor(data[features].values).float()
        self.y = torch.tensor(data[target].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def get_datasets(data_name):
    data_path = 'data/' + data_name
    data_train = pd.read_csv(data_path + '/train.csv')
    data_test = pd.read_csv(data_path + '/test.csv')
    target = DATASETS[data_name]['target']
    features = list(data_train.columns.difference([target]))
    train = CustomDataset(data_train, target, features=features)
    val = CustomDataset(data_test.sample(frac=1.0), target, features=features)
    test = CustomDataset(data_test, target, features=features)

    if data_name not in ['MNIST', 'CelebA']:
        concepts = pd.read_csv(f'{data_path}/concept_groups.csv')
        concept_groups = []
        concept_names = []

        for name, group_df in concepts.groupby('concept'):
            group = []
            for i, row in group_df.iterrows():
                group.append(data_train[features].columns.get_loc(row['feature']))
            print(name, ':', group)
            concept_groups.append(group)
            concept_names.append(name)

        return train, val, test, target, features, concept_groups, concept_names
    else:
        return train, val, test, target, features

def macro_statistics(y_pred, y_true, raw=True):
    if raw:
        y_pred = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)

    # statistics for each class
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    # print(accuracy.shape, precision.shape, recall.shape, f1.shape)

    # macro average
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    macro_accuracy = np.mean(accuracy)

    return macro_accuracy, macro_precision, macro_recall, macro_f1

def adjust_learning_rate(optimizer, lr, epoch, decay=0.1):
    if epoch >= 10:
        lr *= decay
    if epoch >= 20:
        lr *= decay
    if epoch >= 40:
        lr *= decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  

class EarlyStopping(object):
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'Early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss