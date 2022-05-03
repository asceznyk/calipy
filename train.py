import os
import random

import sys
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from utils import *
from model import *

def fit(model, train_loader, valid_loader=None, ckpt_path=None, epochs=10, lr=0.001):  
    def run_epoch(split):
        is_train = split == 'train' 
        model.train(is_train)
        loader = train_loader if is_train else valid_loader

        avg_loss = 0
        avg_mse_percent = 0
        pbar = tqdm(enumerate(loader), total=len(loader))
        for step, batch in pbar:
            batch = [i.to(device) for i in batch]
            imgs, labels = batch
            
            with torch.set_grad_enabled(is_train):  
                preds, loss = model(imgs, labels)

                gt = np.nan_to_num(labels.detach().cpu().numpy()) / max_scale
                mp = preds.detach().cpu().numpy() / max_scale

                err_mse = get_mse(gt, mp)
                zero_mse = get_mse(gt, np.zeros_like(gt))
                mse_score_percent = 100 * np.mean(err_mse)/(np.mean(zero_mse) + 1e-10)

                avg_loss += loss.item() / len(loader)
                avg_mse_percent += mse_score_percent / len(loader)

            if is_train:
                model.zero_grad() 
                loss.backward() 
                optimizer.step()

            pbar.set_description(f"epoch: {e+1}, avg_loss: {avg_loss:.3f}, avg_mse_percent: {avg_mse_percent:.3f}%")     
        return avg_loss

    model.to(device)

    best_loss = float('inf') 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    for e in range(epochs):
        train_loss = run_epoch('train')
        valid_loss = run_epoch('valid') if valid_loader is not None else train_loss

        if ckpt_path is not None and valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), ckpt_path)


def main(args):
    batch_size = args.batch_size
    main_dir = args.main_dir
    
    imgs_train, labels_train, test_file = load_img_vector_pairs(main_dir, ignore_file='2')

    x_test = np.load(open(f'{main_dir}{test_file}.hevc.npy', 'rb'))
    y_test = np.loadtxt(f'{main_dir}{test_file}.txt')

    x_train, x_valid, y_train, y_valid = train_test_split(imgs_train, labels_train, test_size=0.1)

    train_data = CalibData(x_train, y_train)
    valid_data = CalibData(x_valid, y_valid)
    test_data = CalibData(x_test, y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = CalibNet(img_size, label_size)

    '''fit(model, train_loader, valid_loader, ckpt_path='calib.best', epochs=args.epochs)

    model.load_state_dict(torch.load('calib.best'))
    mse_score_percent = calc_percent_error(model, test_loader)

    print(f"YOUR ERROR SCORE ON TEST DATA IS {mse_score_percent:.3f}%") '''

    random_idx = 0   
    print(random_idx)
    single_batch = DataLoader(CalibData(imgs_train[random_idx:random_idx+batch_size], labels_train[random_idx:random_idx+batch_size]), batch_size=batch_size)
    fit(model, single_batch, epochs=args.epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_dir', type=str, help='path to data directory')
    parser.add_argument('--epochs', type=int, help='number of epochs to train the model', default=2)
    parser.add_argument('--batch_size', type=int, help='batch size', default=8)

    options = parser.parse_args()

    print(options)

    main(options)




