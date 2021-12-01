import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

import math
from torch.optim.lr_scheduler import _LRScheduler  # noqa

from sklearn.metrics import roc_auc_score


import nsml
from nsml import DATASET_PATH

USE_CUDA = False


# for All Column running

def build_norm_layer(df: pd.DataFrame, columns: list = None, exclude: list = None, noise: int = 0):
    from collections import OrderedDict
    if columns is not None and exclude is not None:
        raise TypeError
    elif exclude is not None:
        columns = set(df.columns) - set(exclude)
    elif columns is not None:
        columns = set(columns)
    else:
        columns = set(df.columns)
    mean = OrderedDict()
    var = OrderedDict()
    for column_name in df.columns:
        if column_name in columns:
            scaler = StandardScaler(with_std=True, with_mean=True)
            scaler.fit(df[column_name].to_numpy().reshape(-1, 1))
            mean[column_name] = float(scaler.mean_)
            var[column_name] = float(scaler.var_)
        else:
            mean[column_name] = 0.
            var[column_name] = 1.
    mean = list(mean.values())
    var = list(var.values())
    for i in range(noise):
        mean.append(0.)
        var.append(1.)
    mean = torch.Tensor(mean)
    var = torch.Tensor(var)
    return mean, var


class OverSampler:

    def oversample(self, data, label):
        oversampled_data, oversampled_label = SMOTE(random_state=42).fit_resample(data, label)
        return oversampled_data, oversampled_label


def bind_model(model, optimizer=None, scheduler=None):
    def save(path, *args, **kwargs):
        state = {'model': model.state_dict()}
        if optimizer is not None:
            state['optimizer'] = optimizer.state_dict()
        if scheduler is not None:
            state['scheduler'] = scheduler.state_dict()
        torch.save(state, os.path.join(path, 'model.pt'))
        print('Model saved')

    def load(path, *args, **kwargs):
        state = torch.load(os.path.join(path, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        if 'scheduler' in state and scheduler:
            scheduler.load_state_dict(state['scheduler'])
        print('Model loaded')

    def infer(path, **kwargs):
        return inference(path, model)

    nsml.bind(save=save, load=load, infer=infer)  # 'nsml.bind' function must be called at the end.


def inference(path, model, **kwargs):
    model.eval()

    data = Variable(preproc_data(pd.read_csv(path), train=False))
    output_pred_labels = torch.round(torch.sigmoid(model(data)))
    output_pred_labels = output_pred_labels.detach().cpu().numpy()
    output_pred_labels = output_pred_labels.astype('int').reshape(-1).tolist()

    # output format
    # [(step, label), (step, label), ..., (step, label)]
    results = [(step, label) for step, label in enumerate(output_pred_labels)]

    return results


mean_mat = var_mat = None


def preproc_data(data, label=None, train=True, val_ratio=0.2, seed=1234, normalize=False):

    device = torch.device('cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu')

    if train:

        dataset = dict()

        # 범주형
        data['gender_enc'] = np.where(data['gender'] == 'M', 0, 1)

        DROP_COLS = ['CDMID', 'gender', 'date', 'date_E']
        X = data.drop(columns=DROP_COLS).copy()
        X = X.fillna(X.median())
        y = label

        if normalize:
            global mean_mat, var_mat
            mean_mat, var_mat = build_norm_layer(X, exclude=['gender_enc'], noise=1)

        # Oversampling
        oversampler = OverSampler()
        X, y = oversampler.oversample(X, y)

        sLength = len(X['gender_enc'])
        X = X.assign(bias=pd.Series(np.ones(sLength)).values)
        print(sLength)

        # Standard Scaler
        # scaler = StandardScaler()
        # X_cols = X.columns

        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                          stratify=y,
                                                          test_size=val_ratio,
                                                          random_state=seed,
                                                          )

        X_train = torch.as_tensor(X_train.values).float().to(device)
        y_train = torch.as_tensor(y_train.reshape(-1, 1)).float().to(device)
        X_val = torch.as_tensor(X_val.values).float().to(device)
        y_val = torch.as_tensor(y_val.reshape(-1, 1)).float().to(device)

        X = torch.as_tensor(X.values).float().to(device)
        y = torch.as_tensor(y.reshape(-1, 1)).float().to(device)

        dataset['train'] = TensorDataset(X_train, y_train)
        dataset['val'] = TensorDataset(X_val, y_val)
        dataset['all'] = TensorDataset(X, y)

        return dataset

    else:
        data['gender_enc'] = np.where(data['gender'] == 'M', 0, 1)

        # df.loc[:, 'date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        DROP_COLS = ['CDMID', 'gender', 'date', 'date_E']
        data = data.drop(columns=DROP_COLS).copy()
        data = data.fillna(data.median())

        sLength = len(data['gender_enc'])
        data = data.assign(bias=pd.Series(np.ones(sLength)).values)

        X_test = torch.as_tensor(data.values).float().to(device)

        return X_test


class CustomClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomClassifier, self).__init__()
        # first layer = 24 * (2/3) = 16
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.BatchNorm1d(16),
            nn.Sigmoid(),
            nn.Dropout(p=0.5)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(16, 20),
            nn.BatchNorm1d(20),
            nn.Sigmoid(),
            nn.Dropout(p=0.5)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(20, 24),
            nn.BatchNorm1d(24),
            nn.Sigmoid(),
            nn.Dropout(p=0.5)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(24, 20),
            nn.BatchNorm1d(20),
            nn.Sigmoid(),
            nn.Dropout(p=0.5)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(20, 16),
            nn.BatchNorm1d(16),
            nn.Sigmoid(),
            nn.Dropout(p=0.5)
        )

        self.linear = nn.Linear(16, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return self.linear(x)


class MLP(nn.Sequential):

    class Preprocessor(nn.Module):

        def __init__(self, input_size, mean=None, var=None):
            super().__init__()
            self.input_size = int(input_size)
            if mean is None:
                self.register_buffer('mean', torch.zeros(input_size))
            else:
                self.register_buffer('mean', torch.tensor(mean).view(input_size))
            if var is None:
                self.register_buffer('var', torch.ones(input_size))
            else:
                self.register_buffer('var', torch.tensor(var).view(input_size))

        def forward(self, x):
            return torch.div(torch.sub(x, self.mean), self.var)

        def extra_repr(self):
            return "size: %s" % self.input_size

    def __init__(
            self,
            input_size,
            output_size,
            preproc_mean=None,
            preproc_var=None,
            channels=(16, 20, 24, 20, 16),
            dropout=0.4
    ):
        super().__init__()
        features = input_size
        self.preprocessor = self.Preprocessor(input_size, preproc_mean, preproc_var)
        for i, channel in enumerate(channels, start=1):
            self.add_module(
                "layer%s" % i,
                nn.Sequential(
                    nn.Linear(features, channel),
                    nn.BatchNorm1d(channel),
                    nn.ELU(),
                    nn.Dropout(p=dropout)
                )
            )
            features = channel
        self.classifier = nn.Linear(features, output_size)
        # layers.append(nn.Tanh())

    def forward(self, x):
        if self.training:
            x[:, -1] = torch.rand_like(x[:, -1])
        else:
            x[:, -1] = torch.zeros_like(x[:, -1])
        return super().forward(x)


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (
                        1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class ClassificationLoss(nn.BCEWithLogitsLoss):

    def __init__(self):
        super().__init__(reduction='none')

    def forward(self, logit, target):
        loss = super().forward(logit, target)
        ground_truth = torch.eq(torch.round(torch.sigmoid(logit)), target)
        return torch.mean(torch.where(ground_truth, loss, loss * 2))


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--batch_size', type=int, default=128)
    # args.add_argument('--val_ratio', type=int, default=0.2)
    args.add_argument('--val_ratio', type=int, default=0.3)
    args.add_argument('--lr', type=float, default=0.1)
    args.add_argument('--input_size', type=int, default=22)
    args.add_argument('--epochs', type=int, default=1000)
    args.add_argument('--no_cuda', action='store_true')

    config = args.parse_args()

    def seed_all(seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.determinstic = False

    # seed_all()

    USE_CUDA = False
    device = torch.device('cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu')

    time_init = time.time()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = MLP(config.input_size+1, 1).to(device)

    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = ClassificationLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = MultiStepLR(optimizer, milestones=[100, 200, 300, 400, 500, 600, 700, 800], gamma=0.5)

    # for cosine scheduler
    optimizer = optim.Adam(model.parameters(), lr=0)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=100, T_mult=1, eta_max=0.1, T_up=10, gamma=0.5)

    # nsml.bind() should be called before nsml.paused()
    bind_model(model, optimizer=optimizer)

    # DONOTCHANGE: They are reserved for nsml
    # Warning: Do not load data before the following code!
    # test mode
    if config.pause:
        nsml.paused(scope=locals())

    # training mode
    if config.mode == 'train':
        data_path = DATASET_PATH + '/train/train_data'
        label_path = DATASET_PATH + '/train/train_label'

        raw_data = pd.read_csv(data_path)
        raw_labels = np.loadtxt(label_path, dtype=np.int16)

        # dataset = preproc_data(raw_data, raw_labels, train=True, val_ratio=0.3, seed=1234)

        normalize = True

        dataset = preproc_data(raw_data, raw_labels, train=True, val_ratio=0.2, seed=1234, normalize=normalize)

        if normalize:
            preproc_layer = model.preprocessor
            preproc_layer.mean = mean_mat
            preproc_layer.var = var_mat

        # train_dl = DataLoader(dataset['train'], config.batch_size, shuffle=True)
        # val_dl = DataLoader(dataset['val'], config.batch_size, shuffle=False)
        train_dl = DataLoader(dataset['all'], config.batch_size, shuffle=True)
        val_dl = DataLoader(dataset['all'], config.batch_size, shuffle=False)
        time_dl_init = time.time()
        print('Time to dataloader initialization: ', time_dl_init - time_init)

        min_val_loss = np.inf
        loss_list = []
        score_list = []

        for epoch in range(config.epochs):
            # train model
            running_loss = 0.
            num_runs = 0
            model.train()
            total_length = len(train_dl)
            for iter_idx, (data, labels) in enumerate(train_dl):
                data = Variable(data)
                labels = Variable(labels)

                output_pred = model(data)
                loss = loss_fn(output_pred, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_runs += 1

                # get current lr
                opt_params = optimizer.state_dict()['param_groups'][0]
                step = epoch * total_length + iter_idx

                nsml.report(
                    epoch=epoch + int(config.iteration),
                    epoch_total=config.epochs,
                    iter=iter_idx,
                    iter_total=total_length,
                    batch_size=config.batch_size,
                    train__loss=running_loss / num_runs,
                    step=step,
                    lr=opt_params['lr'],
                    scope=locals()
                )
            scheduler.step()

            train_loss = running_loss / num_runs

            print(f"[Epoch {epoch}] Loss: {train_loss}")

            with torch.no_grad():

                # test model with validation data
                model.eval()
                running_loss = 0.
                num_runs = 0

                pd = []
                gt = []

                correct, total = 0, 0
                for data, labels in val_dl:
                    data = Variable(data)
                    labels = Variable(labels)

                    output_pred = model(data)
                    loss = loss_fn(output_pred, labels)

                    running_loss += loss.item()
                    num_runs += 1

                    output_pred_labels = torch.round(torch.sigmoid(output_pred))

                    total += labels.size(0)
                    correct += (output_pred_labels == labels).sum().item()

                    pd.append(output_pred)
                    gt.append(labels)

                pd = torch.cat(pd, dim=0)
                gt = torch.cat(gt, dim=0)

                val_loss = running_loss / num_runs
                loss_list.append(val_loss)

                accuracy = 100.0 * float(correct) / float(total)
                score = roc_auc_score(gt.data.cpu().numpy(), pd.data.cpu().numpy())
                score_list.append(score)

                print(f"[Validation] Loss: {running_loss / num_runs}")
                print('Accuracy: %f %%, ROC: %f' % (accuracy, score))

                nsml.report(
                    summary=True,
                    epoch=epoch,
                    epoch_total=config.epochs,
                    train__loss=train_loss,
                    val__loss=val_loss,
                    step=(epoch + 1) * total_length,
                    lr=opt_params['lr']
                )

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                nsml.save(epoch)
            elif score > np.array(score_list).max():
                nsml.save(epoch)
            elif epoch % 10 == 0:
                nsml.save(epoch)

        print(f"Best Val Loss Epoch : {np.array(loss_list).argmin()} / Loss : {np.array(loss_list).min()}")
        print(f"Best Score Epoch : {np.array(score_list).argmax()} / Score : {np.array(score_list).max()}")
        final_time = time.time()
        print("Time to dataloader initialization: ", time_dl_init - time_init)
        print("Time spent on training :", final_time - time_dl_init)
        print("Total time: ", final_time - time_init)

        print("Done")
