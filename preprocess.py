import os
import sys
import glob
import cv2

import numpy as np

from utils import img_size

def main(videos_dir, resize, ext='*.hevc'):
    os.chdir(videos_dir)
    for video_path in glob.glob(ext): 
        frames = []
        cap = cv2.VideoCapture(video_path)
        ret = True
        while ret:
            ret, img = cap.read() 
            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if int(resize):
                    img = cv2.resize(img, dsize=(img_size[2], img_size[1]), interpolation=cv2.INTER_CUBIC)
                frames.append(img)
                
        np.save(f'{video_path}.npy', np.stack(frames, axis=0))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])



