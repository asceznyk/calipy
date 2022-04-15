import os
import sys
import glob
import cv2
import numpy as np

def load_img_vector_pairs(_dir):
    os.chdir(_dir)
    labels = []
    imgs = []
    for file in glob.glob('*'):
        if file.endswith('.txt'):
            labels.append(np.loadtxt(f'{_dir}/{file}'))
        elif file.endswith('.npy'):
            imgs.append(np.load(open(f'{_dir}/{file}' , 'rb')))

    print(imgs.shape)
    return np.stack(imgs), np.stack(labels)






