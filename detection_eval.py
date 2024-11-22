import argparse
import numpy as np
import cv2
from detector import detect_tag
import json
from vis_utils import draw_tag, drawAxesWithPose
import os
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_params(file_path):
    try:
        with open(file_path, 'r') as file:
            params = json.load(file)
            index = np.array(params["index"])
            sig = np.array(params["dict_sig"])
            world_loc = np.array(params["dict_world_loc"])
        return index, sig, world_loc
    except Exception as e:
        logging.error(f"Error loading parameters: {e}")
        raise

def detect(input_path, output_path):
    missed_detection = 0
    index, dict_sig, dict_world_loc = load_params('lasrtag_dictionary.json')

    image_files = [f for f in os.listdir(input_path) if f.endswith('.png')]

    for image_name in image_files:
        img_path = os.path.join(input_path, image_name)
        img = cv2.imread(img_path)

        if img is None:
            logging.warning(f"Image {img_path} not found or unable to read.")
            continue

        result, count = detect_tag(img, dict_sig)
        out_img = draw_tag(img, result)
        out_img = drawAxesWithPose(out_img, result, dict_world_loc)

        if count == 0:
            logging.info(f"Missed detection in {image_name}")
            missed_detection += 1

        out_name = os.path.join(output_path, image_name)
        cv2.imwrite(out_name, out_img)

    return missed_detection

if __name__ == "__main__":
    setup_logging()

    input_path = '/home/ravikt/repositories/lasrtag/results/quarter_res/input_images'
    output_path = '/home/ravikt/repositories/lasrtag/results/quarter_res/detections'

    missed_detections = detect(input_path, output_path)
    logging.info(f"Total Missed Detections: {missed_detections}")
