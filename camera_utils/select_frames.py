import numpy as np
import cv2
import shutil

idx = np.random.randint(0, 4618, 60)

#print(idx[8])

for i in idx:
    src = "dataset/calibration/refraction_cam_calib/frame{}.png".format(i)
    dst = "dataset/calibration/selected_frames/frames{}.png".format(i)
    shutil.move(src, dst)
