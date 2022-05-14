import pandas as pd
import numpy as np
import pdb
import warnings
from glob import glob
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F
from model.get_model import get_model

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(file_list):
    test_x = torch.tensor(np.load("./Spectrogram/test_x.npy", allow_pickle = True), dtype=torch.float32)
    test_x = test_x.reshape(-1, 1, test_x.shape[1], test_x.shape[2])
    test_ds = TensorDataset(test_x)
    test_loader = DataLoader(test_ds, batch_size = 30, shuffle=False)

    pred_sum = torch.zeros(test_x.shape[0], 6)

    for file_name in tqdm(file_list):
        with torch.no_grad():
            result_dict = torch.load(f"./model/result/{file_name}.pt")
            model_hparam, state_dict = result_dict['hparam'], result_dict['weight']

            model_class = get_model(model_hparam['model'])
            model = model_class(**model_hparam).to(device)
            model.load_state_dict(state_dict)
            model.eval()

            pred = []

            for x in test_loader:
                x = x[0].to(device)
                pred.append(model(x).detach().cpu())

            pred = torch.cat(pred, axis = 0)
            pred_sum += pred
    
    pred_sum /= len(file_list)
    pred_sum = F.softmax(pred_sum).numpy()
    make_submission_file(pred_sum)

def make_submission_file(pred):
    test_file_list = glob("./open/test/*.wav")
    idx_list = list( map(lambda x : int(x.split("\\")[1].split(".")[0]), test_file_list) )
    submission = pd.read_csv('./open/sample_submission.csv')
    
    for idx, prd in zip(idx_list, pred):
        
        submission.iloc[idx-1, 1:] = prd

    now = time.localtime()
    submission.to_csv(f'./submission/{now.tm_mon:02d}{now.tm_mday:02d} {now.tm_hour:02d}{now.tm_min:02d}.csv', index=False)

if __name__ == "__main__":
    file_list = ["0514 " + x for x in ["1012", "1030", "1047", "1104", "1121"]]
    test(file_list)