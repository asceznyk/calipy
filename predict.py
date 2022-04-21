import os
import sys
import argparse
import cv2

import numpy as np

import torch
import torch.nn as nn

from utils import *
from model import *

def main(args):
    model = CalibNet(img_size, label_size)
    if args.ckpt_path:
        model.load_state_dict(torch.load(args.ckpt_path))

    cap = cv2.VideoCapture(args.video_path)
    ret = True
    f = 0
    while ret:
        ret, img = cap.read() 
        if ret:
            img = cv2.resize(img, dsize=(img_size[2], img_size[1]), interpolation=cv2.INTER_CUBIC)
            img = torch.from_numpy(img/255.0).float()
            img = img.view(*img_size).unsqueeze(0)
            predictions = model(img)

            print(predictions, f)
            
            f += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='path to input video')
    parser.add_argument('--ckpt_path', default='', type=str, help='path to trained model')
    options = parser.parse_args()

    print(options)

    main(options)










