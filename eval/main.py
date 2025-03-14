import cv2
import numpy as np
import os

num = 1100
missed_detection = 0
task_name = 'moving_target'
home_dir = "/home/ravikt/lasr/icra_test/dataset/aruco/"

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

for i in range(num):

   # load image
    image_path=os.path.join(home_dir,"input_images",task_name, "frame{}.png".format(i))
    image = cv2.imread(image_path)

    corners, ids, rejected = detector.detectMarkers(image)
   
    
       

    if len(corners) > 0:
	    # flatten the ArUco IDs list
        ids = ids.flatten()
	    # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw the bounding box of the ArUCo detection
            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            # draw the ArUco marker ID on the image
            cv2.putText(image, str(markerID),
                (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
            print("[INFO] ArUco marker ID: {}".format(markerID))
            # show the output imag
    else:
        missed_detection = missed_detection+1
    if not os.path.exists(os.path.join('results', task_name)):
        os.makedirs(os.path.join('results', task_name))
    out_path = os.path.join("results",task_name,"frame{}.png".format(i))
    cv2.imwrite(out_path, image)
            

detection = num - missed_detection
detection_rate = (detection/num)*100
print("Total Missed Detection:", missed_detection)
print("Detection Rate: ", detection_rate)
