import os
import sys
import glob
import cv2
import numpy as np

def main(videos_dir, ext='.hevc'):
    os.chdir(videos_dir)
    for video_path in glob.glob(ext): 
        frames = []
        cap = cv2.VideoCapture(video_path)
        ret = True
        while ret:
            ret, img = cap.read() 
            if ret:
                img = cv2.resize(img, dsize=(266, 200), interpolation=cv2.INTER_CUBIC)
                frames.append(img)
                
        np.save(f'{video_path}.npy', np.stack(frames, axis=0))

if __name__ == '__main__':
    main(sys.argv[1])



