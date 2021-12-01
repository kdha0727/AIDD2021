# -*- coding: utf-8 -*-
# +
# PCA
# -

import argparse
import sys
import os
import time

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric

from imblearn.over_sampling import SMOTE, RandomOverSampler
from smote_variants import MSMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


# -

import nsml
from nsml import DATASET_PATH


def bind_model(model, optimizer=None):
    def save(path, *args, **kwargs):
        torch.save(model, os.path.join(path, 'model.pt'))
        print('Model saved')

    def load(path, *args, **kwargs):
        net = torch.load(os.path.join(path, 'model.pt'))
        model.__dict__.update(net.__dict__)
        print('Model loaded')

    # 추론
    def infer(path, **kwargs):
        return inference(path, model)

    nsml.bind(save=save, load=load, infer=infer)  # 'nsml.bind' function must be called at the end.



# 추론
def inference(path, model, **kwargs):
    model.eval()

    data = Variable(preproc_data(pd.read_csv(path), train=False))
    output_pred_labels = torch.round(model(data))
    output_pred_labels = output_pred_labels.detach().numpy()
    output_pred_labels = output_pred_labels.astype('int').reshape(-1).tolist()

    # output format
    # [(step, label), (step, label), ..., (step, label)]
    results = [(step, label) for step, label in enumerate(output_pred_labels)]

    return results


import math
from torch.optim.lr_scheduler import _LRScheduler  # noqa


class OverSampler:  # SMOTE, RandomOverSampler
    def __init__(self):
        pass

    def oversample(self, data, label):
        oversampled_data, oversampled_label = SMOTE(random_state=42).fit_resample(data, label)
        return oversampled_data, oversampled_label

# +
# 데이터 전처리


def preproc_data(data, label=None, train=True, val_ratio=0.2, seed=1234):
    if train:
        dataset = dict()

        # 성별 ['M', 'F'] -> [0, 1]로 변환
        data['gender_enc'] = np.where(data['gender'] == 'M', 0, 1)

        # 날짜 datetime으로 변환
        # df.loc[:, 'date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        DROP_COLS = ['CDMID', 'gender', 'date', 'date_E']
        X = data.drop(columns=DROP_COLS).copy()
        X = X.fillna(X.median())
        y = label
        
#         SELECT_COLS = "GGT TG BMI Cr FBG ALT HDL BUN HbA1c CrCl Alb AST age Wt SBP PR ALP gender_enc".split()
#         SELECT_COLS = "age Wt SBP PR FBG TG LDL BUN CrCl ALT ALP".split()
        SELECT_COLS = "GGT TG BMI Cr FBG ALT HDL BUN HbA1c CrCl Alb AST age gender_enc".split()
        X = X[SELECT_COLS]

        # Oversampling
        oversampler = OverSampler()
        X_, y_ = X.copy(), y.copy()
        X, y = oversampler.oversample(X, y)

        X = X.assign(bias=pd.Series(np.ones(len(X))).values)
        X_ = X_.assign(bias=pd.Series(np.ones(len(X_))).values)

        # Standard Scaler
        # scaler = StandardScaler()
        # X_cols = X.columns


        X_train, X_val_, y_train, y_val_ = train_test_split(X, y,
                                                          stratify=y,
                                                          test_size=val_ratio,
                                                          random_state=seed,
                                                          )

        X_train = torch.as_tensor(X_train.values).float()
        y_train = torch.as_tensor(y_train.reshape(-1, 1)).float()
        X_val = torch.as_tensor(X_.values).float()
        y_val = torch.as_tensor(y_.reshape(-1, 1)).float()

        X = torch.as_tensor(X.values).float()
        y = torch.as_tensor(y.reshape(-1, 1)).float()

        dataset['train'] = TensorDataset(X_train, y_train)
        dataset['val'] = TensorDataset(X_val, y_val)
        dataset['all'] = TensorDataset(X, y)

        return dataset

    else:
        # 성별 ['M', 'F'] -> [0, 1]로 변환
        data['gender_enc'] = np.where(data['gender'] == 'M', 0, 1)

        # 날짜 datetime으로 변환
        # df.loc[:, 'date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        DROP_COLS = ['CDMID', 'gender', 'date', 'date_E']
        data = data.drop(columns=DROP_COLS).copy()
        data = data.fillna(data.median())
        
#         SELECT_COLS = "GGT TG BMI Cr FBG ALT HDL BUN HbA1c CrCl Alb AST age Wt SBP PR ALP gender_enc".split()
#         SELECT_COLS = "age Wt SBP PR FBG TG LDL BUN CrCl ALT ALP".split()
        SELECT_COLS = "GGT TG BMI Cr FBG ALT HDL BUN HbA1c CrCl Alb AST age gender_enc".split()
        data = data[SELECT_COLS]

        sLength = len(data)
        data = data.assign(bias=pd.Series(np.ones(sLength)).values)

        X_test = torch.as_tensor(data.values).float()

        return X_test


# xbnet 으로 transfer 해보기
# lr scheduler 를 다른 걸 한번 써보기 (Cosine)
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    
    from XBNet.training_utils import training,predict
    from XBNet.models import XBNETClassifier
    from XBNet.run import run_XBNET

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--batch_size', type=int, default=32)
    # args.add_argument('--val_ratio', type=int, default=0.2)
    args.add_argument('--val_ratio', type=int, default=0.3)
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--input_size', type=int, default=14)  # 22
    args.add_argument('--epochs', type=int, default=300)
    
    argv = [] if ('In' in globals() and 'Out' in globals()) else None
    
    config = args.parse_args(argv)

    time_init = time.time()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    model = XBNETClassifier.lazy_construct()

    # training mode
    if config.mode == 'train':
        data_path = DATASET_PATH + '/train/train_data'
        label_path = DATASET_PATH + '/train/train_label'

        raw_data = pd.read_csv(data_path)
        raw_labels = np.loadtxt(label_path, dtype=np.int16)

        # dataset = preproc_data(raw_data, raw_labels, train=True, val_ratio=0.3, seed=1234)
        dataset = preproc_data(raw_data, raw_labels, train=True, val_ratio=0.2, seed=1234)
        
        # train_dl = DataLoader(dataset['train'], config.batch_size, shuffle=True)
        
        train_dl = DataLoader(dataset['all'], config.batch_size, shuffle=True)
        val_dl = DataLoader(dataset['all'], config.batch_size, shuffle=False)
        time_dl_init = time.time()
        print('Time to dataloader initialization: ', time_dl_init - time_init)
        
        model.lazy_init(
            dataset['all'].tensors[0].numpy(), 
            dataset['all'].tensors[1].numpy(), 
            6, 3,
            input_through_cmd=True,
            inputs_for_gui=[
                14 + 1, 16,
                16, 20,
                20, 24,
                24, 20,
                20, 16,
                16, 1
            ]
        )
        model.sequential.add_module(str(len(model.sequential)), nn.Sigmoid())

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # nsml.bind() should be called before nsml.paused()
    bind_model(model)

    # DONOTCHANGE: They are reserved for nsml
    # Warning: Do not load data before the following code!
    # test mode
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        m,acc, lo, val_ac, val_lo = run_XBNET(
            # x_train,x_test,y_train,y_test,
            dataset['all'].tensors[0].numpy(), 
            dataset['all'].tensors[0].numpy(), 
            dataset['all'].tensors[1].numpy(), 
            dataset['all'].tensors[1].numpy(), 
            
            model,criterion,optimizer,
            epochs=config.epochs,batch_size=config.batch_size
        )
        nsml.report(
            summary=True,
            epoch=config.epochs,
            epoch_total=config.epochs,
        )
        nsml.save(0)

        final_time = time.time()
        print("Time to dataloader initialization: ", time_dl_init - time_init)
        print("Time spent on training :", final_time - time_dl_init)
        print("Total time: ", final_time - time_init)

        print("Done")

# # 


