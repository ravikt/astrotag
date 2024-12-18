import os
import cv2
from pupil_apriltags import Detector

task_name = 'moving_target'
home_dir = "/home/ravikt/lasr/icra_test/dataset/apriltag/"
num=1100
missed_detection=0


at_detector = Detector(
   families="tagStandard41h12",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)


# image_path=os.path.join(home_dir,"input_images",task_name, "frame{}.png".format(0))
# image = cv2.imread(image_path)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# results = at_detector.detect(gray)

for i in range(num):

    image_path=os.path.join(home_dir,"input_images",task_name, "frame{}.png".format(i))
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = at_detector.detect(gray)

    if not results:
        missed_detection = missed_detection+1

    for r in results:
    # extract the bounding box (x, y)-coordinates for the AprilTag
    # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("[INFO] tag family: {}".format(tagFamily))
        # show the output image after AprilTag detection

    if not os.path.exists(os.path.join('results', task_name)):
        os.makedirs(os.path.join('results', task_name))
    out_path = os.path.join("results",task_name,"frame{}.png".format(i))
    cv2.imwrite(out_path, image)

detection = num - missed_detection
detection_rate = (detection/num)*100
print("Total Missed Detection:", missed_detection)
print("Detection Rate: ", detection_rate)