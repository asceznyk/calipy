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

def load_img_vector_pairs(_dir, ignore_idx=0):
    os.chdir(_dir)
    labels = []
    imgs = []
    for i, file in enumerate(sorted(glob.glob('*'))):
        if i != ignore_idx:
            if file.endswith('.txt'):
                print(file)
                labels.extend(np.loadtxt(f'{_dir}/{file}'))
            elif file.endswith('.npy'):
                print(file)
                imgs.extend(np.load(open(f'{_dir}/{file}' , 'rb')))
        else:
            ignore_file = file.replace('.txt', '').replace('.npy', '').replace('.hevc')

    return np.array(imgs), np.array(labels), ignore_file

def get_mse(gt, test):
  test = np.nan_to_num(test)
  return np.mean(np.nanmean((gt - test)**2, axis=0))





