import argparse
import json
import numpy as np
import cv2
from detector import create_tag_dict, detect_tag
from vis_utils import draw_tag, drawAxesWithPose, drawPoseCube
import matplotlib.pyplot as plt
import os


# def detect(input_path, output_path):
# num = 100
index=[]
sig=[]
missed_detection = 0
with open('output/new_dictionary.json', 'r') as file:
    params = json.load(file)
    # world_loc = np.array(params["world_points"])
    for key, value in params.items():   
        
        sig.append(value)
    dict_sig=np.array(sig)


# Process all images in test_images folder
test_folder = 'test_images'
# test_folder = '/home/ravikt/Desktop/videos/spacecraft/lasrtag/lasrtag_static_100cm'
for filename in os.listdir(test_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(test_folder, filename)
        output_path = os.path.join('results', filename)
        # print(input_path)
        # Read image and detect tags
        img = cv2.imread(input_path)
        result,count = detect_tag(img, dict_sig)
        # count = detect_tag(img, dict_sig)

        if count==0:
            print("missed")
            missed_detection = missed_detection + 1
        else:
            print("detected: ", count)
        # Draw detection results and save
        out_img = draw_tag(img, result)
        # out_img = drawAxesWithPose(out_img, result)
        out_img = drawPoseCube(img, result)
        os.makedirs('results', exist_ok=True)
        cv2.imwrite(output_path, out_img)
        