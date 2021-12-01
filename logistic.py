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

import nsml
from nsml import DATASET_PATH


def bind_model(model, optimizer=None):
    def save(path, *args, **kwargs):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(path, 'model.pt'))
        print('Model saved')

    def load(path, *args, **kwargs):
        state = torch.load(os.path.join(path, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Model loaded')

    # 추론
    def infer(path, **kwargs):
        return inference(path, model)
    
    nsml.bind(save=save, load=load, infer=infer) # 'nsml.bind' function must be called at the end.


# 추론
def inference(path, model, **kwargs):
    model.eval()
    
    data = Variable(preproc_data(pd.read_csv(path), train=False))
    output_pred_labels = torch.round(torch.sigmoid(model(data)))
    output_pred_labels = output_pred_labels.detach().numpy()
    output_pred_labels = output_pred_labels.astype('int').reshape(-1).tolist()

    # output format
    # [(step, label), (step, label), ..., (step, label)]
    results = [(step, label) for step, label in enumerate(output_pred_labels)]
    
    return results


# 데이터 전처리
def preproc_data(data, label=None, train=True, val_ratio=0.2, seed=1234):
    if train:
        dataset = dict()

        # NaN 값 0으로 채우기
        data = data.fillna(0)

        # 성별 ['M', 'F'] -> [0, 1]로 변환
        data['gender_enc'] = np.where(data['gender'] == 'M', 0, 1)

        # 날짜 datetime으로 변환
        # df.loc[:, 'date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        DROP_COLS = ['CDMID', 'gender', 'date', 'date_E']
        X = data.drop(columns=DROP_COLS).copy()
        y = label

        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                          stratify=y, 
                                                          test_size=val_ratio,
                                                          random_state=seed,
                                                          )

        X_train = torch.as_tensor(X_train.values).float()
        y_train = torch.as_tensor(y_train.reshape(-1, 1)).float()
        X_val = torch.as_tensor(X_val.values).float()
        y_val = torch.as_tensor(y_val.reshape(-1, 1)).float()

        dataset['train'] = TensorDataset(X_train, y_train)
        dataset['val'] = TensorDataset(X_val, y_val)

        return dataset
    
    else:
        # NaN 값 0으로 채우기
        data = data.fillna(0)

        # 성별 ['M', 'F'] -> [0, 1]로 변환
        data['gender_enc'] = np.where(data['gender'] == 'M', 0, 1)

        # 날짜 datetime으로 변환
        # df.loc[:, 'date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        DROP_COLS = ['CDMID', 'gender', 'date', 'date_E']
        data = data.drop(columns=DROP_COLS).copy()

        X_test = torch.as_tensor(data.values).float()

        return X_test


# 모델
class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--val_ratio', type=int, default=0.2)
    args.add_argument('--lr', type=float, default=0.1)
    args.add_argument('--input_size', type=int, default=22)
    args.add_argument('--epochs', type=int, default=30)
    config = args.parse_args()

    time_init = time.time()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = LogisticRegression(config.input_size, 1)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

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
        dataset = preproc_data(raw_data, raw_labels, train=True, val_ratio=0.2, seed=1234)
        
        train_dl = DataLoader(dataset['train'], config.batch_size, shuffle=True)
        val_dl = DataLoader(dataset['val'], config.batch_size, shuffle=False)
        time_dl_init = time.time()
        print('Time to dataloader initialization: ', time_dl_init - time_init)

        min_val_loss = np.inf
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
                step = epoch*total_length+iter_idx

                nsml.report(
                    epoch=epoch+int(config.iteration),
                    epoch_total=config.epochs,
                    iter=iter_idx,
                    iter_total=total_length,
                    batch_size=config.batch_size,
                    train__loss=running_loss / num_runs,
                    step=step,
                    lr=opt_params['lr'],
                    scope=locals()
                )
            
            print(f"[Epoch {epoch}] Loss: {running_loss / num_runs}")

            # test model with validation data
            model.eval()
            running_loss = 0.
            num_runs = 0
            for data, labels in val_dl:
                data = Variable(data)
                labels = Variable(labels)

                output_pred = model(data)
                loss = loss_fn(output_pred, labels)

                running_loss += loss.item()
                num_runs += 1
            
            print(f"[Validation] Loss: {running_loss / num_runs}")

            nsml.report(
                summary=True,
                epoch=epoch,
                epoch_total=config.epochs,
                val__loss=running_loss / num_runs,
                step=(epoch+1) * total_length,
                lr=opt_params['lr']
            )

            if (running_loss < min_val_loss) or (epoch % 10 == 0):
                nsml.save(epoch)

        final_time = time.time()
        print("Time to dataloader initialization: ", time_dl_init - time_init)
        print("Time spent on training :",final_time - time_dl_init)
        print("Total time: ", final_time - time_init)
        
        print("Done")