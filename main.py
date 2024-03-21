import numpy as np
import cv2
from detector import create_tag_dict, detect_tag, find_squares
import json
from utils import draw_tag, drawAxesWithPose
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

for i in range(14):

    # print('Length of dict_sig:', len(dict_sig))

    img = cv2.imread("../../../dataset/astrotag_multiple/frame{}.png".format(i))
    # thresh, cands = find_squares(img)

    result = detect_tag(img, dict_sig)

    out_img = draw_tag(img, result)

    out_img = drawAxesWithPose(img, result, dict_world_loc)
    cv2.imwrite("results/astrotag_multi_chaser/frame{}.png".format(i),out_img)
