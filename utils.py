import os
import glob
import numpy as np

import torch

from torch.utils.data import Dataset, DataLoader

class CalibData(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __getitem__(self, i):
        img = torch.tensor(self.imgs[i]/255.0).float()
        label = self.labels[i]
        return img, label

def load_img_vector_pairs(_dir):
    os.chdir(_dir)
    labels = []
    imgs = []
    for file in sorted(glob.glob('*')):
        if file.endswith('.txt'):
            print(file)
            labels.extend(np.loadtxt(f'{_dir}/{file}'))
        elif file.endswith('.npy'):
            print(file)
            imgs.extend(np.load(open(f'{_dir}/{file}' , 'rb')))

    return np.array(imgs), np.array(labels)





