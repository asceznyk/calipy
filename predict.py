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
    video_path = args.video_path

    model = CalibNet(img_size, label_size)
    if args.ckpt_path:
        model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()

    cap = cv2.VideoCapture(video_path)
    ret = True
    f = 0

    ext = os.path.splitext(video_path)[1]
    file = open(video_path.replace(ext, 'pred.txt'), 'w')
    while ret:
        ret, img = cap.read() 
        if ret:
            img = cv2.resize(img, dsize=(img_size[2], img_size[1]), interpolation=cv2.INTER_CUBIC)
            with torch.no_grad():
                img = torch.from_numpy(img/255.0).float()
                img = img.view(*img_size).unsqueeze(0)
                angles, _ = model(img)
                angles = angles.detach().cpu().numpy()

            file.write(np.array2string(angles, separator=' ')[1:-1]+'\n')
            f += 1

    file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='path to input video')
    parser.add_argument('--ckpt_path', default='', type=str, help='path to trained model')
    options = parser.parse_args()

    print(options)

    main(options)










