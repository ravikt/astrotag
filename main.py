import argparse
import numpy as np
import cv2
from detector import create_tag_dict, detect_tag, find_squares
import json
from utils import draw_tag, drawAxesWithPose
import matplotlib.pyplot as plt
import os


def detect(input_path, output_path):
    num = 100
    missed_detection = 0
    with open('lasrtag_dictionary.json', 'r') as file:
        params = json.load(file)

        index = np.array(params["index"])
        sig   = np.array(params["dict_sig"])
        world_loc = np.array(params["dict_world_loc"])


    # print(sig[0][3])
    # print(world_loc[0][3])
    dict_sig = sig
    dict_world_loc = world_loc

    for i in range(num):

        # print('Length of dict_sig:', len(dict_sig))
        image_name = os.path.join(input_path, "frame{}.png".format(i))
        # img = cv2.imread("../../../dataset/astrotag_multiple/frame{}.png".format(i))
        img = cv2.imread(image_name)

        # thresh, cands = find_squares(img)

        result, count = detect_tag(img, dict_sig)

        out_img = draw_tag(img, result)

        out_img = drawAxesWithPose(img, result, dict_world_loc)

        if count==0:
            print("missed")
            missed_detection = missed_detection + 1


        out_name = os.path.join(output_path, "frame{}.png".format(i))
        cv2.imwrite(out_name, out_img)

    return missed_detection
        # cv2.imwrite("results/astrotag_multi_chaser/frame{}.png".format(i),out_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script for running lasrtag detector")
    parser.add_argument("-p", "--path", type=str, help="path to lasrtag images")

    parser.add_argument("-o", "--output", type=str, help="path to output images")
    args = parser.parse_args()
    input_path = args.path
    output_path = args.output

    m=detect(input_path, output_path)

    print("Total Missed Detection:", m)

