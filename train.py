import os
import sys
import argparse
import numpy as np

from utils import *

def main(train_dir):
    imgs, labels = load_img_vector_pairs(train_dir)
    print(imgs.shape, labels.shape)

if __name__ == '__main__':
    main(sys.argv[1])






