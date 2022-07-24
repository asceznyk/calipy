import os
import random

import sys
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from utils import *
from model import *

def fit(model, optimizer, train_loader, valid_loader=None, ckpt_path=None, epochs=10, lr=0.001, log_preds=[0,1]): 
    print(f'the learning rate chosen: {lr}')
    
    if type(log_preds) == int:
        log_preds = [log_preds, log_preds]

    def run_epoch(split, log=0):
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
                avg_loss += loss.item() / len(loader)
                avg_mse_percent += mse_zero_percent(np.nan_to_num(labels.detach().cpu().numpy()),  preds.detach().cpu().numpy()) / len(loader)
                
                if log:
                    print('-'*40)
                    print(f'predictions for frames ->')
                    print(preds)
                    print(f'labels for frames ->')
                    print(torch.nan_to_num(labels))
                    print('-'*40)

            if is_train:
                model.zero_grad() 
                loss.backward() 
                optimizer.step()

            pbar.set_description(f"epoch: {e+1}, avg_loss: {avg_loss:.6f}, avg_mse_percent: {avg_mse_percent:.3f}%") 

        return avg_loss

    model.to(device)

    best_loss = float('inf') 
    train_losses, valid_losses = [], []
    for e in range(epochs):
        train_loss = run_epoch('train', log_preds[0])
        valid_loss = run_epoch('valid', log_preds[1]) if valid_loader is not None else train_loss
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        if ckpt_path is not None and valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), ckpt_path)
            
    return train_losses, valid_losses

def main(main_dir, epochs=50, batch_size=4, learning_rate=1e-4, single_batch=0, zero_input=0, pretrained_weights=''):
    torch.manual_seed(0)

    img_paths, labels = load_img_path_labels(f'{challenge_path}labeled')
    train_data, valid_data, test_data = split_data(img_paths, labels)
    labels = np.nan_to_num(labels)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = CalibConvNet(img_size, label_size)
    
    print(model)
    print(f'total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
    if pretrained_weights != '':
        print(f'loading pretrained model from {pretrained_weights}.. ')
        load_pretrained_model(model, pretrained_weights)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    if single_batch:
        random_idx = 2401
        labels = fill_zeros_previous(labels)
        if not zero_input:
            single_data = CalibData(img_paths[random_idx:random_idx+batch_size], labels[random_idx:random_idx+batch_size]) 
            single_batch = DataLoader(single_data, batch_size=batch_size)
        else:
            print(f'all inputs to model are zeros, checking training results..')
            single_batch = DataLoader(DummyData(img_size, labels[random_idx: random_idx+batch_size]), batch_size=batch_size)

        train_losses, valid_losses = fit(model, optimizer, single_batch, epochs=epochs, lr=learning_rate, log_preds=1) 
    else:
        train_losses, valid_losses = fit(model, optimizer, train_loader, valid_loader, epochs=epochs, lr=learning_rate, log_preds=0, ckpt_path='calibnet.best')
        plt.plot(valid_losses)
        
    plt.plot(train_losses)
    plt.show()

    plt.savefig('loss_plot.png')
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_dir', type=str, help='path to data directory')
    parser.add_argument('--epochs', type=int, help='number of epochs to train the model', default=10)
    parser.add_argument('--batch_size', type=int, help='batch size', default=4)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=1e-4)
    parser.add_argument('--single_batch', type=int, default=0)
    parser.add_argument('--zero_input', type=int, default=0)
    parser.add_argument('--pretrained_weights', type=str, default='calibnet.best', help='the saved set of model weights to train/finetune on..')

    options = parser.parse_args()

    print(options)

    main(options)




