import numpy as np
import cv2
from lasr_detector import loadMarkerDictionary, detectAruco, find_squares
from lasr_utils import drawArucos, drawAxesWithPose
import matplotlib.pyplot as plt

marker = cv2.imread("marker_aruco.png")
dict_sig, dict_world_loc = loadMarkerDictionary(marker, 49)


for i in range(1400):
    img = cv2.imread("aruco/new/frame{}.jpg".format(i))
    thresh, cands = find_squares(img)


    result = detectAruco(img, dict_sig, dict_world_loc)


    out_img = drawArucos(img, result)

    out_img = drawAxesWithPose(img, result, dict_world_loc)
    cv2.imwrite("results/aruco_pose/frame{}.png".format(i),out_img)

# plt.imshow(out_img)
# plt.show()