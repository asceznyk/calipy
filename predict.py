import os
import sys
import glob
import argparse
import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torch.nn as nn

from utils import *
from model import *

def main(args):
    data_dir = args.data_dir
    out_dir = args.out_dir
    ext = args.ext
    make_video = args.make_video
    
    if args.model_type == 'conv':
        model = CalibConvNet(img_size, label_size)
    else:
        model = CalibResNet(img_size, label_size)

    if args.ckpt_path != '':
        model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    os.chdir(data_dir)

    for video_path in glob.glob(ext):
        cap = cv2.VideoCapture(video_path)
        ret = True
        f = 0
        file = open(out_dir + '/' + video_path.replace(ext[1:], '.txt'), 'w')

        fig = plt.figure()  
        frames = [] 

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
                
                if make_video:
                    _img = cv2.putText(
                        _img, 
                        angle_str, 
                        (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.25, 
                        255
                    )
                    _img = plt.imshow(_img, animated=True)
                    frames.append([_img])

                file.write(angle_str+'\n')
                f += 1
        
        if make_video:
            ani = animation.ArtistAnimation(
                fig, frames, interval=40, blit=True, repeat_delay=1000
            )
            ani.save(f'{video_path}.mp4')

        file.close()
        print(f'finished predicting on video: {video_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='path to input videos directory')
    parser.add_argument('--out_dir', type=str, help='path to output predictions directory')
    parser.add_argument('--ckpt_path', default='', type=str, help='path to trained model')
    parser.add_argument('--model_type', default='conv', type=str, help='type of model')
    parser.add_argument('--ext', default='*.hevc', type=str, help='format of video (ext)')
    parser.add_argument('--make_video', type=int, default=0, help='make video if needed')
    options = parser.parse_args()

    print(options)

    main(options)



