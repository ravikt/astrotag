import stag
import cv2
import numpy as np
import os

task_name="results"
input_dir = 'stag_100cm'
# home_dir = "results"
# input_dir = os.path.join(home_dir, "input_images", task_name)
missed_detection = 0

# Get all PNG images from the input directory
image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
total_images = len(image_files)

for image_file in image_files:
   # load image
   image_path = os.path.join(input_dir, image_file)
   image = cv2.imread(image_path)

   # detect markers
   (corners, ids, rejected_corners) = stag.detectMarkers(image, 19)
   if not ids:
      print(f"Missed detection in {image_file}")
      missed_detection = missed_detection + 1
   
   # draw detected markers with ids
   stag.drawDetectedMarkers(image, corners, ids)

   # draw rejected quads without ids with different color
   stag.drawDetectedMarkers(image, rejected_corners, border_color=(255, 0, 0))

   # save resulting image
   output_dir = os.path.join('results', task_name)
   if not os.path.exists(output_dir):
      os.makedirs(output_dir)
   out_path = os.path.join(output_dir, image_file)
   
   cv2.imwrite(out_path, image)

detection = total_images - missed_detection
detection_rate = (detection/total_images)*100
print(f"Total Images Processed: {total_images}")
print(f"Total Missed Detection: {missed_detection}")
print(f"Detection Rate: {detection_rate:.2f}%")
