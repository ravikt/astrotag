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

# img = cv2.imread('test.png')

# result = detect_tag(img, dict_sig)

# out_img = draw_tag(img, result)

# cv2.imwrite('output.png', out_img)
# Process all images in test_images folder
test_folder = 'test_images'
for filename in os.listdir(test_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(test_folder, filename)
        output_path = os.path.join('results', filename)
        
        # Read image and detect tags
        img = cv2.imread(input_path)
        result = detect_tag(img, dict_sig)
        
        # Draw detection results and save
        out_img = draw_tag(img, result)
        os.makedirs('results', exist_ok=True)
        cv2.imwrite(output_path, out_img)