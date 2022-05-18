import pandas as pd
import numpy as np
import pdb
import warnings
import time
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F
from model.get_model import get_model

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s      %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler('log.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def train(model_hparam, pre_trained=None):
    train_x = torch.tensor(np.load("./Spectrogram/train_x_5.npy", allow_pickle = True), dtype=torch.float32)
    train_y = torch.tensor(np.load("./Spectrogram/train_y_5.npy", allow_pickle = True), dtype=torch.float32)
    train_x = train_x.reshape(-1, 1, train_x.shape[1], train_x.shape[2])

    bs, epochs, iteration = 32, 30, 1
    skf = StratifiedKFold(n_splits=5, random_state=64, shuffle=True)
 
    for train_idx, val_idx in skf.split(train_x, train_y):
        train_ds = TensorDataset(train_x[train_idx], train_y[train_idx])
        train_loader = DataLoader(dataset=train_ds, batch_size=bs, num_workers=2)
        val_ds = TensorDataset(train_x[val_idx], train_y[val_idx])
        val_loader = DataLoader(dataset=val_ds, batch_size=bs, num_workers=2)
        
        model_class = get_model(model_hparam['model'])
        model = model_class(**model_hparam).to(device)

        if pre_trained != None:
            result_dict = torch.load(f"./result/model/{pre_trained}.pt")
            assert model_hparam == result_dict['hparam'], "Your model must be same with pre trained model"
            model_hparam, state_dict = result_dict['hparam'], result_dict['weight']
            model.load_state_dict(state_dict)

        now = time.localtime()
        model_name = f'{now.tm_mon:02d}{now.tm_mday:02d} {now.tm_hour:02d}{now.tm_min:02d}'
        os.mkdir(f"./result/tensorboard/{model_name}")
        writer = SummaryWriter(f"./result/tensorboard/{model_name}/runs")

        opt = torch.optim.Adam(model.parameters(), lr =1e-3)
        lrs = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = epochs)
        loss_fn = nn.CrossEntropyLoss()
        best = 9999
        
        for epoch in range(epochs):
            start = time.time()
            model.train()
            train_loss = 0
            pred_list, true_list = [], []
            for _, (x,y) in enumerate(train_loader):
                x = x.to(device)
                y = y.type(torch.LongTensor).to(device)
                
                opt.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                opt.step()
                
                train_loss += loss.item()
                pred_list += F.softmax(pred).argmax(1).detach().cpu().numpy().tolist()
                true_list += y.detach().cpu().numpy().tolist()
                
            lrs.step()
            train_acc = accuracy_score(true_list, pred_list)

            with torch.no_grad():
                model.eval()
                val_loss = 0
                pred_list, true_list = [], []
                for _, (x, y) in enumerate(val_loader):
                    x = x.to(device)
                    y = y.type(torch.LongTensor).to(device)
                    
                    pred = model(x)
                    loss = loss_fn(pred, y)
                    
                    val_loss += loss.item()
                    pred_list += F.softmax(pred).argmax(1).detach().cpu().numpy().tolist()
                    true_list += y.detach().cpu().numpy().tolist()
                    
            val_acc = accuracy_score(true_list, pred_list)
            
            if val_loss/len(val_loader) < best:
                best = val_loss/len(val_loader)
                torch.save({'hparam':model_hparam, 'weight':model.state_dict(), 'score' : best}, f'./result/model/{model_name}.pt')
            
            logger.info(f'===================== Epoch : {epoch+1}/{epochs}    time : {time.time()-start:.0f}s =====================')
            logger.info(f'TRAIN -> loss : {train_loss/len(train_loader):.5f}     accuracy : {train_acc:.5f}')
            logger.info(f'VALID -> loss : {val_loss/len(val_loader):.5f}     accuracy : {val_acc:.5f}    best : {best:.5f}\n\n')

            writer.add_scalar(f"loss/train", train_loss/len(train_loader), epoch)
            writer.add_scalar(f"loss/val", val_loss/len(val_loader), epoch)
            writer.add_scalar(f"acc/train", train_acc, epoch)
            writer.add_scalar(f"acc/val", val_acc, epoch)

if __name__ == "__main__":
    model_hparam = {"model" : "Private1st", "N" : 4}
    train(model_hparam)