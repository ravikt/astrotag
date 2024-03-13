import numpy as np
import cv2
from detector import create_tag_dict, detect_tag, find_squares
from utils import draw_tag, drawAxesWithPose
import matplotlib.pyplot as plt

marker = cv2.imread("astrotag.png")
dict_sig, dict_world_loc = create_tag_dict(marker, 700)

print(dict_world_loc)

for i in range(300):
    img = cv2.imread("/home/ravikt/lasr/fiducial/markers/astrotag/offset_sample/frame{}.png".format(i))
    thresh, cands = find_squares(img)


    result = detect_tag(img, dict_sig, dict_world_loc)


    out_img = draw_tag(img, result)

    out_img = drawAxesWithPose(img, result, dict_world_loc)
    cv2.imwrite("results/frame{}.png".format(i),out_img)

# plt.imshow(out_img)
# plt.show()