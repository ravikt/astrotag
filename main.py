import argparse
import numpy as np
import cv2
from detector import create_tag_dict, detect_tag, find_squares
import json
from vis_utils import draw_tag, drawAxesWithPose
import matplotlib.pyplot as plt
import os


# def detect(input_path, output_path):
# num = 100
index=[]
sig=[]
missed_detection = 0
with open('grs/output/message_pairs24.json', 'r') as file:
    params = json.load(file)
    for key, value in params.items():
        # print(key,value["message"])
        index.append(key)
        sig.append(value["message"])
    index=np.array(index)
    dict_sig=np.array(sig)

img = cv2.imread('test24.jpg')

result = detect_tag(img, dict_sig)

out_img = draw_tag(img, result)

cv2.imwrite('output.jpg', out_img)