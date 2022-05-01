import os
import argparse

import glob
import numpy as np

import torch

from torch.utils.data import Dataset, DataLoader

max_scale = 10
img_size = (3, 200, 266)
label_size = (2,)

class CalibData(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, i):
        img = torch.from_numpy(self.imgs[i]/255.0).float()
        size = img.size()
        label = torch.from_numpy(self.labels[i] * max_scale).float()
        return img.view(size[2], size[0], size[1]), label

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

def calc_percent_error(model, loader, gt):
    model.eval()

    mp = []
    for imgs, _ in loader:
        preds, _ = model(imgs)
        mp.extend(preds.detach().cpu().numpy())
    mp = np.array(mp)

    err_mse = get_mse(gt, mp)
    zero_mse = get_mse(gt, np.zeros_like(gt))
    mse_score_percent = 100 * np.mean(err_mse)/(np.mean(zero_mse) + 1e-10)

    print(mse_score_percent)
    return mse_score_percent




