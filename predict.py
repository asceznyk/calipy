import os
import sys
import glob
import argparse
import cv2

import numpy as np

import torch
import torch.nn as nn

from utils import *
from model import *

def main(args):
    data_dir = args.data_dir
    ext = args.ext

    model = CalibNet(img_size, label_size)
    if args.ckpt_path != '':
        model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()

    os.chdir(data_dir)
    
    print(glob.glob(ext))

    for video_path in glob.glob(ext):
        cap = cv2.VideoCapture(video_path)
        ret = True
        f = 0
        file = open(video_path.replace(ext, '.pred.txt'), 'w')

        while ret:
            ret, img = cap.read() 
            if ret:
                img = cv2.resize(img, dsize=(img_size[2], img_size[1]), interpolation=cv2.INTER_CUBIC)
                with torch.no_grad():
                    img = torch.from_numpy(img/255.0).float()
                    img = img.view(*img_size).unsqueeze(0)
                    angles, _ = model(img)
                    angles = angles.detach().cpu().numpy()[0]
                    angles /= max_scale 

                file.write(np.array2string(angles, separator=' ') [1:-1]+ '\n')
                f += 1

        file.close()
        print(f'finished predicting on video: {video_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='path to input videos directory')
    parser.add_argument('--ckpt_path', default='', type=str, help='path to trained model')
    parser.add_argument('--ext', default='*.hevc', type=str, help='format of video (ext)')
    options = parser.parse_args()

    print(options)

    main(options)



