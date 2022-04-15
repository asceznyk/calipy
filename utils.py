import os
import sys
import glob
import cv2
import numpy as np

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






