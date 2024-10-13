import numpy as np
import cv2
from detector import detect_tag
import json
from vis_utils import drawAxesWithPose
import matplotlib.pyplot as plt


with open('marker_dictionary.json', 'r') as file:
    params = json.load(file)

    index = np.array(params["index"])
    sig   = np.array(params["dict_sig"])
    world_loc = np.array(params["dict_world_loc"])


# print(sig[0][3])
# print(world_loc[0][3])
dict_sig = sig
dict_world_loc = world_loc

for i in range(1):

    # print('Length of dict_sig:', len(dict_sig))

    img = cv2.imread("frame{}.png".format(i))

    result = detect_tag(img, dict_sig)

    # out_img = draw_tag(img, result)

    out_img = drawAxesWithPose(img, result, dict_world_loc, use_lie_algebra=True)
    cv2.imwrite("out_frame{}.png".format(i),out_img)
