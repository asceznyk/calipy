import os
import sys
import numpy as np

import matplotlib.pyplot as plt 

from utils import *

def main(train_dir):
    imgs, labels = load_img_vector_pairs(train_dir)
    plt.imshow(imgs[1000], interpolation='nearest')
    plt.savefig('01000.png')

if __name__ == '__main__':
    main(sys.argv[1])






