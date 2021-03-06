import os
import argparse

import glob
import numpy as np

import torch
import torchvision.transforms as T

from torch.utils.data import Dataset, DataLoader

max_scale = 1
img_size = (3, 200, 266)
label_size = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CalibData(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels
        self.transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor()
        ])

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, i):
        img = self.transform(self.imgs[i])
        label = torch.from_numpy(self.labels[i] * max_scale).float()
        return img, label 

def load_img_vector_pairs(_dir, ignore_file='0'):
    os.chdir(_dir)

    labels = []
    imgs = []
    for file in sorted(glob.glob('*')):
        if not file.startswith(ignore_file):
            if file.endswith('.txt'):
                print(file)
                labels.extend(np.loadtxt(f'{_dir}/{file}'))
            elif file.endswith('.npy'):
                print(file)
                imgs.extend(np.load(open(f'{_dir}/{file}' , 'rb')))

    return np.array(imgs), np.array(labels), ignore_file

def get_mse(gt, test):
  test = np.nan_to_num(test)
  return np.mean(np.nanmean((gt - test)**2, axis=0))

def calc_percent_error(model, loader):
    model.eval()

    mp = []
    gt = []
    for imgs, labels in loader:
        preds, _ = model(imgs.to(device))
        mp.extend(preds.detach().cpu().numpy() / max_scale)
        gt.extend(labels.detach().cpu().numpy() / max_scale)

    gt = np.array(gt)
    mp = np.array(mp) 

    print(gt)
    print(mp)

    err_mse = get_mse(gt, mp)
    zero_mse = get_mse(gt, np.zeros_like(gt))
    mse_score_percent = 100 * np.mean(err_mse)/(np.mean(zero_mse) + 1e-10)

    print(err_mse, zero_mse)
    return mse_score_percent




