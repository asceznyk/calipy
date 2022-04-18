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
    img_size = img.size()
    label_size = label.size()
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    model = CalibNet(img_size, label_size)

    x, y = next(iter(loader))
    print(x.size())

    out, loss = model(x, y)

    print(out.size(), loss)
    print(out)

    x, y = next(iter(loader))
    print(x.size())

    out, loss = model(x, y)

    print(out.size(), loss)
    print(out)

if __name__ == '__main__':
    main(sys.argv[1])






