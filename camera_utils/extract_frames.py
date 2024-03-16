import cv2

vidcap = cv2.VideoCapture('dataset/calibration/cam_calib.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("dataset/calibration/cam_calib/frame%d.png" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
