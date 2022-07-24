import os
import glob
import math
import numpy as np

import torch
import torchvision.transforms.functional as VF

from torch.utils.data import Dataset, DataLoader

max_scale = 1
img_size = (3, 188, 250)
label_size = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CalibData(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img = VF.resize(VF.to_tensor(Image.open(self.img_paths[i])), (img_size[1], img_size[2]))
        label = torch.from_numpy(self.labels[i] * max_scale).float()
        
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
class DummyData(Dataset):
    def __init__(self, img_size, labels):
        self.mats = torch.zeros((labels.shape[0], *img_size))
        self.labels = labels

    def __len__(self):
        return self.mats.shape[0]

    def __getitem__(self, i):
        mat = self.mats[i]
        label = torch.from_numpy(self.labels[i] * max_scale).float()
        return mat, label
    
def view_angle_images(data, start, end):
    show([data[i][0] for i in range(start, end)])
    print([data[i][1] for i in range(start, end)])
    
def load_img_path_labels(input_dir):
    labels = []
    img_paths = []
    for file in sorted([f for f in os.listdir(input_dir) if f.endswith('txt')]):
        labels.append(np.loadtxt(f'{input_dir}/{file}'))
        for l in range(1, len(labels[-1])+1):
            img_paths.append(f"{input_dir}/{file.replace('.txt', '')}_{l}.jpg")
    return np.array(img_paths), np.vstack(labels)

def get_mse(gt, test):
    test = np.nan_to_num(test)
    return np.mean(np.nanmean((gt - test)**2, axis=0))

def to_radians(deg):
    return deg * math.pi / 180

def mse_zero_percent(gt, mp, convert=0):
    if convert:
        gt = to_radians(gt)
        mp = to_radians(mp)
        
    err_mse = get_mse(gt, mp)
    zero_mse = get_mse(gt, np.zeros_like(gt))
    
    return 100 * (err_mse / (zero_mse if zero_mse > 0 else 1.25e-3))

def fill_zeros_previous(arr):
    for i, r in enumerate(arr):
        if r.sum() == 0 and i > 0:
            arr[i] = arr[i-1]
    return arr
            
def remove_zero_labels(x, y):
    y = y[np.all(y != 0, axis=1)]
    x = x[np.where(np.any(y != 0, axis=1))[0]]
    return x, y
            
def split_data(img_paths, labels, split=0.90, transform=None, non_zero_labels=1, remove_nans=1):
    labels = np.nan_to_num(labels)
    
    if non_zero_labels:
        if remove_nans:
            img_paths, labels = remove_zero_labels(img_paths, labels)
        else:
            labels = fill_zeros_previous(labels)
 
    x_train, x_test, y_train, y_test = train_test_split(img_paths, labels, test_size=(1.0 - split), random_state=42)
    train_size = int(split * x_train.shape[0])
    x_valid, y_valid, x_train, y_train = x_train[train_size:], y_train[train_size:], x_train[:train_size], y_train[:train_size]

    train_data = CalibData(x_train, y_train, transform=transform)
    valid_data = CalibData(x_valid, y_valid)
    test_data = CalibData(x_test, y_test)
    
    return train_data, valid_data, test_data

def load_pretrained_model(model, weights_path):
    model.load_state_dict(torch.load(weights_path))
    return model




