#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xlrd
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.optimize as spop
# кульбак лейбнер
from scipy.special import kl_div
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
# Импорт openpyxl
import openpyxl
from openpyxl import load_workbook
import random
from scipy.special import rel_entr
import torch
from torch.nn.functional import softmax
from torch.nn.functional import kl_div
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from multiprocessing import cpu_count
from pathlib import Path

# In[120]:
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

seed = 1
np.random.seed(seed)
torch.cuda.set_device(0)  # if you have more than one CUDA device

# In[121]:


# ROOT = Path.cwd().parent/'input'
# SAMPLE = ROOT/'sample_submission.csv'
# TRAIN = ROOT/'X_train.csv'
# TARGET = ROOT/'y_train.csv'
# TEST = ROOT/'X_test.csv'

ID_COLS = ['series_id', 'index_bank']

# In[122]:


# x_trn = pd.read_csv('C:/Users/mikhe/Downloads/X_train.csv', usecols=x_cols.keys(), dtype=x_cols)
# x_tst = pd.read_csv('C:/Users/mikhe/Downloads/X_test.csv', usecols=x_cols.keys(), dtype=x_cols)
# y_trn = pd.read_csv('C:/Users/mikhe/Downloads/y_train.csv', usecols=y_cols.keys(), dtype=y_cols)
x_trn = pd.read_csv('C:/Users/mikhe/Credit_score/X_train_quad.csv')
x_tst = pd.read_csv('C:/Users/mikhe/Credit_score/X_test_quad.csv')
y_trn = pd.read_csv('C:/Users/mikhe/Credit_score/y_train_quad.csv')
y_tst = pd.read_csv('C:/Users/mikhe/Credit_score/y_test_quad.csv')


# In[123]:


def create_datasets(X, y, test_size=0.2, dropcols=ID_COLS, time_dim_first=False):
    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)
    X_grouped = create_grouped_array(X)
    if time_dim_first:
        X_grouped = X_grouped.transpose(0, 2, 1)
    X_train, X_valid, y_train, y_valid = train_test_split(X_grouped, y_enc,
                                                          test_size=0.1)  # делим трейн данные на трей и валидационные
    X_train, X_valid = [torch.tensor(arr, dtype=torch.float32) for arr in (X_train, X_valid)]
    y_train, y_valid = [torch.tensor(arr, dtype=torch.long) for arr in (y_train, y_valid)]
    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)
    return train_ds, valid_ds, enc


def create_datasets_test(X, y, dropcols=ID_COLS, time_dim_first=False):
    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)
    X_grouped = create_grouped_array(X)
    if time_dim_first:
        X_grouped = X_grouped.transpose(0, 2, 1)
    X_test = torch.tensor(X_grouped, dtype=torch.float32)
    y_test = torch.tensor(y_enc, dtype=torch.long)
    test_ds = TensorDataset(X_test, y_test)
    return test_ds, enc


def create_grouped_array(data, group_col='series_id', drop_cols=ID_COLS):
    X_grouped = np.row_stack([
        group.drop(columns=drop_cols).values[None]
        for _, group in data.groupby(group_col)])
    return X_grouped


def create_test_dataset(X, drop_cols=ID_COLS):
    X_grouped = np.row_stack([
        group.drop(columns=drop_cols).values[None]
        for _, group in X.groupby('series_id')])
    X_grouped = torch.tensor(X_grouped.transpose(0, 2, 1)).float()
    y_fake = torch.tensor([0] * len(X_grouped)).long()
    return TensorDataset(X_grouped, y_fake)


def create_loaders(train_ds, valid_ds, bs=16, jobs=0):
    train_dl = DataLoader(train_ds, bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, bs, shuffle=False)
    return train_dl, valid_dl


def create_loader_test(test_ds, bs=16, jobs=0):
    test_dl = DataLoader(test_ds, bs, shuffle=True)
    return test_dl


def accuracy(output, target):
    return (output.argmax(dim=1) == target).float().mean().item()


def create_test_dataset_fake(X, drop_cols=ID_COLS):
    X_grouped = np.row_stack([
        group.drop(columns=drop_cols).values[None]
        for _, group in X.groupby('series_id')])
    X_grouped = torch.tensor(X_grouped.transpose(0, 2, 1)).float()
    y_fake = torch.tensor([0] * len(X_grouped)).long()
    return TensorDataset(X_grouped, y_fake)


# In[124]:


class CyclicLR(_LRScheduler):

    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]


# In[125]:


def cosine(t_max, eta_min=0):
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min) * (1 + np.cos(np.pi * t / t_max)) / 2

    return scheduler


# In[126]:


# визуализация изменения learning rate
n = 100
sched = cosine(n)
lrs = [sched(t, 1) for t in range(n * 4)]


# In[127]:


# class LSTMClassifier(nn.Module):
#     """Very simple implementation of LSTM-based time-series classifier."""
#
#     # инициализирует атрибуты класса и определяет слои LSTM (rnn) и полносвязанный слой (fc).
#     def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
#         super().__init__()
#         self.hidden_dim = hidden_dim  # кол-во скрытого пр-ва (1)
#         self.layer_dim = layer_dim  # кол-во скрытых слоев (4)
#         self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)  # инициализируется слой LSTM
#         self.fc = nn.Linear(hidden_dim,
#                             output_dim)  # создает линейный слой (fully connected) для преобразования выходных данных LSTM в прогнозируемые классы или значения
#         self.batch_size = None
#         self.dropout = nn.Dropout(0.2)
#         self.hidden = None
#
#     def forward(self, x):
#         h0, c0 = self.init_hidden(x)  # инициализация скрытых состояний h0 и c0 с помощью метода init_hidden(x).
#         out, (hn, cn) = self.rnn(x, (h0, c0))  # прямой проход модели
#         out = self.dropout(out)
#         out = self.fc(out[:, -1,
#                       :])  # выбираются выходы LSTM только для последнего временного шага каждой последовательности в батче
#         # Результатом является тензор прогнозов классов с размерностью (batch_size, output_dim)
#         return out
#
#     def init_hidden(self, x):
#         h0 = torch.zeros(self.layer_dim, x.size(0),
#                          self.hidden_dim)  # количество слоев, размер батча, размер скрытого состояния
#         c0 = torch.zeros(self.layer_dim, x.size(0),
#                          self.hidden_dim)  # количество слоев, размер батча, размер скрытого состояния
#         return [t.cuda() for t in (h0, c0)]

class RNNClassifier(nn.Module):
    """Simple implementation of RNN-based time-series classifier."""

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True)  # Замена LSTM на RNN
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.dropout = nn.Dropout(0.1)
        self.hidden = None

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, hn = self.rnn(x, h0)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        return h0.cuda()
# In[130]:


print('Preparing datasets')
trn_ds, val_ds, enc = create_datasets(x_trn, y_trn['rating'])

# In[131]:


bs = 16
print(f'Creating data loaders with batch size: {bs}')
trn_dl, val_dl = create_loaders(trn_ds, val_ds, bs, jobs=cpu_count())

# In[133]:
hidden_list = [8, 16]
layer_list = [1]
accuracy_list = []
rmse_list = []
description_list = []
for hidden in hidden_list:
    for layer in layer_list:
        for iter_i in range(10):
            input_dim = 171  # размерность входная
            hidden_dim = hidden  # размерность скрытого пространства (ht)
            layer_dim = layer  # кол-во скрытых слоев
            output_dim = 8  # размерность выходная (количество признаков)
            # seq_dim = 128    # длина последовательности

            lr = 0.0005  # скорость обучения
            n_epochs = 1000  # количество эпох
            iterations_per_epoch = len(trn_dl)  # итераций в эпоху
            best_acc = 0  # лучший accuracy
            patience, trials = 200, 0  # сколько эпох держит без повышения accuracy, счетчик

            # model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)  # создаем класс LSTM
            model = RNNClassifier(input_dim, hidden_dim, layer_dim, output_dim)  # создаем класс LSTM

            model = model.cuda()  # кидаем на CUDA
            criterion = nn.CrossEntropyLoss()  # критерий (или функция потерь) для задач классификации
            opt = torch.optim.RMSprop(model.parameters(), lr=lr)  # выбор оптимизатора
            sched = CyclicLR(opt, cosine(t_max=iterations_per_epoch * 2, eta_min=lr / 100))

            print('Start model training')

            # Для каждой эпохи делаем
            for epoch in range(1, n_epochs + 1):
                # Для счетчика, набора батчей в трейн дата лоадере
                for i, (x_batch, y_batch) in enumerate(trn_dl):
                    model.train()  # перевод модели в режим обучения
                    x_batch = x_batch.to(DEVICE)  # кидаем в CUDA
                    y_batch = y_batch.to(DEVICE)  # кидаем в CUDA
                    sched.step()  # обновление расписания scheduler для обновления скорости обучения
                    opt.zero_grad()  # обнуление градиента
                    out = model(x_batch)  # прямой проход по модели (на вход x_batch)
                    loss = criterion(out, y_batch)  # смотрим чо по лоссу
                    loss.backward()  # обратное распространение ошибки (вычисляем градиент по всем параметрам модели)
                    opt.step()  # оптимизатором меняем веса для уменьшения функции потерь

                model.eval()  # перевод модели в режим оценки
                correct, total = 0, 0  # переменные правильно прдсказанные / всего

                # Для набора батчей в валидационном дата лоадере
                for x_val, y_val in val_dl:
                    x_val, y_val = [t.to(DEVICE) for t in (x_val, y_val)]  # переносим все элементы в CUDA
                    out = model(x_val)  # прямой проход по модели (на вход x_val)
                    preds = F.log_softmax(out, dim=1).argmax(dim=1)  # логарифм от софтмакса и по нему максимальный
                    total += y_val.size(0)  # увеличиваем тотал на батч у_вал
                    correct += (preds == y_val).sum().item()  # увеличиваем коррект если правильно

                acc = correct / total  # метрика
                # Для каждой 5 эпохи
                if epoch % 5 == 0:
                    print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Acc.: {acc:2.2%}')
                # Если текущая метрика выше лучшей то обновляем лучшую
                if acc > best_acc:
                    trials = 0
                    best_acc = acc
                    torch.save(model.state_dict(), 'best.pth')
                    print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
                else:
                    trials += 1
                    if trials >= patience:
                        print(f'Early stopping on epoch {epoch}')
                        break

            model.load_state_dict(torch.load('best.pth'))  # сохраненные параметры модели, полученные в процессе обучения

            # In[71]:

            model.eval()  # перевод модели в режим оценки

            # <b>ТУТ МЫ ТЕСТИМ АКУРАСИ НА ТЕСТОВЫХ ДАННЫХ</b>

            test_dl = DataLoader(create_test_dataset_fake(x_tst), batch_size=16, shuffle=False)

            test = []
            print('Predicting on test dataset')
            for batch, _ in test_dl:
                batch = batch.permute(0, 2, 1)
                out = model(batch.cuda())
                y_hat = F.log_softmax(out, dim=1).argmax(dim=1)
                test += y_hat.tolist()

            y_pred = enc.inverse_transform(test)
            y_real = y_tst['rating'].values

            correct, total = 0, 0  # переменные правильно прeдсказанные / всего
            total = len(y_pred)
            # Для набора батчей в валидационном дата лоадере
            for i in range(len(y_pred)):
                if y_pred[i] == y_real[i]:
                    correct += 1
            acc = correct / total  # метрика


            # Рассчет квадратов разницы между предсказаниями и фактическими значениями
            squared_diffs = (y_real - y_pred) ** 2

            # Рассчет среднеквадратичного отклонения (RMSE)
            rmse = torch.sqrt(torch.mean(torch.tensor(squared_diffs, dtype=torch.float32)))

            # print(f'Acc.: {acc:2.2%}')
            # print('RMSE:', rmse.item())
            # print(f'Hidden dim {hidden}, layer {layer} :: Acc.:{acc:2.2%}, RMSE:{rmse.item()}')
            description = (iter_i, hidden, layer, acc, rmse.item())
            description_list.append(description)

            # accuracy_list.append(acc)
            # rmse_list.append(rmse.item())
            # print(f'Predicted: {y_pred}')
            # print(f'Real     : {y_real}')

            torch.cuda.empty_cache()

for desc in description_list:
    print(f'Iter {desc[0]}, Hidden dim {desc[1]}, layer {desc[2]} :: Acc.:{desc[3]:2.2%}, RMSE:{desc[4]:.2f}')
