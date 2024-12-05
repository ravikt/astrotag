import stag
import cv2
import numpy as np
import os

num = 100
missed_detection = 0
task_name = 'fixed_100cm_24lux'
home_dir = "/home/ravikt/lasr/icra_test/dataset/stag/"
for i in range(num):

   # load image
   image_path=os.path.join(home_dir,"input_images",task_name, "frame{}.png".format(i))
   image = cv2.imread(image_path)

   # detect markers
   (corners, ids, rejected_corners) = stag.detectMarkers(image, 19)
   if not ids:
      print("missed")
      missed_detection = missed_detection + 1
   # draw detected markers with ids
   stag.drawDetectedMarkers(image, corners, ids)

   # draw rejected quads without ids with different color
   stag.drawDetectedMarkers(image, rejected_corners, border_color=(255, 0, 0))

   # save resulting image
   if not os.path.exists(os.path.join('results', task_name)):
        os.makedirs(os.path.join('results', task_name))
   out_path = os.path.join("results",task_name,"frame{}.png".format(i))
   
   cv2.imwrite(out_path, image)

detection = num - missed_detection
detection_rate = (detection/num)*100
print("Total Missed Detection:", missed_detection)
print("Detection Rate: ", detection_rate)
