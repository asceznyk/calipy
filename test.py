import os
import sys
import glob
import cv2

import numpy as np

import torch
import torch.nn as nn

from utils import *
from model import *

def main(args):
    cap = cv2.VideoCapture(args.video_path)
    ret = True
    f = 0

    model = CalibNet(img_size, label_size)
    if args.ckpt_path != '':
        model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()
   
    while ret:
        ret, img = cap.read()
        if ret:
            _img = cv2.resize(img, dsize=(img_size[2], img_size[1]), interpolation=cv2.INTER_CUBIC)
            with torch.no_grad():
                img = torch.from_numpy(_img/255.0).float()
                img = img.view(*img_size).unsqueeze(0)
                angles, _ = model(img)
                angles = angles.detach().cpu().numpy()[0]
                angles /= max_scale

            angle_str = np.array2string(angles, separator=' ')[1:-1]
            _img = cv2.putText(
                _img, 
                angle_str, 
                (0,0), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.25, 
                255
            )

            cv2.imwrite('angles_img.png', _img)
            f += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='path to input video')
    parser.add_argument('--ckpt_path', default='', type=str, help='path to trained model')
    options = parser.parse_args()

    print(options)

    main(options)




