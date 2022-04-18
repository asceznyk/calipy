import os
import sys
import numpy as np

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from utils import *
from model import *

def main(train_dir):
    batch_size = 8
    
    imgs, labels = load_img_vector_pairs(train_dir) 

    data = CalibData(imgs, labels)
    img, label = data[0]
    img_size = img.shape[1:]
    label_size = label.shape[1:]
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    x, _ = next(iter(loader))
    print(x.size())

    model = CalibNet(img_size, label_size)

    out = model(x)
    print(out.size())

if __name__ == '__main__':
    main(sys.argv[1])






