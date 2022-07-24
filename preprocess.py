import os
import sys
import glob
import cv2

import numpy as np

from utils import img_size

def main(videos_dir, resize=0):
    for video_path in [file for file in os.listdir(videos_dir) if file.endswith('.hevc')]: 
        video_path = f'{videos_dir}/{video_path}'
        cap = cv2.VideoCapture(video_path)
        ret = True
        f = 1
        while ret:
            ret, img = cap.read()
            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if resize:
                    img = cv2.resize(img, dsize=(img_size[2], img_size[1]), interpolation=cv2.INTER_AREA)
                cv2.imwrite(f"{video_path.replace('.hevc', '')}_{f}.jpg", img)
            f += 1

if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))



