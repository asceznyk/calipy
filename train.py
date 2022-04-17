import os
import sys
import numpy as np

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from utils import *

def main(train_dir):
    imgs, labels = load_img_vector_pairs(train_dir)

if __name__ == '__main__':
    main(sys.argv[1])






