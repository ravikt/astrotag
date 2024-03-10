import cv2
import json
import numpy as np

def draw_grid(img, row, col):

    cellW = int(img.shape[1]/col)
    cellH = int(img.shape[0]/row)

    # print(cellH, cellW)
    for i in range(int(row)):
        y = i*cellH
        cv2.line(img, (0,y), (img.shape[1], y), (255,0,0), 1)

    for j in range(int(col)):
        x = j*cellW
        cv2.line(img, (x,0), (x, img.shape[0]), (255,0,0), 1)

    return img

def equalSig(sig1, sig2, allowedMisses=0):
    misses = 0
    for i in range(len(sig1)):
        if sig1[i] != sig2[i]:
            misses = misses + 1
    print('Misses: ', misses)
    return misses<=allowedMisses

def drawArucos(img, res):

    '''
    The function drawContour takes the contour list as a python array not np array
    '''
    print(len(res))
    for i in range(len(res["ar_index"])):
    
        corner = res["ar_corners"][i]
        corner = np.asarray(corner).reshape((4,2)).astype(int)
        cv2.drawContours(img, [corner], -1, (255, 0, 0))
        org = (corner[0][0], corner[0][1])

        idx = res["ar_index"][i]
        cv2.putText(img, str(idx), org, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0))

    return img

def drawAxesWithPose(img, res, dict_world_loc):

    with open('camera_intrinsic_chaser.json', 'r') as camera_data:
        camera_params = json.load(camera_data)

    mtx = np.array(camera_params["camera_matrix"], dtype="double")
    dist= np.array(camera_params["distortion_coefficients"])

    axis = np.float32([[0,0,0], [25,0,0], [0, 25, 0], [0, 0, -25]])

    # Convert the points from marker coordinate system to image coordinate system 

    for i in range(len(res["ar_index"])):

        corner = res["ar_corners"][i]
        idx = res["ar_index"][i]

        corner = np.asarray(corner).reshape((4,2)).astype('float32')
        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv2.solvePnP(dict_world_loc[idx], corner, mtx, dist)

        # draw axis on the marker
        # project points from marker coordinate system in the image coordinate system 

        img_pts, _ = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        # print(tuple(img_pts[0].ravel()))
        cv2.line(img, tuple(img_pts[0].ravel().astype(int)), tuple(img_pts[1].ravel().astype(int)), (255,0,0), 2, cv2.LINE_AA) # Red
        cv2.line(img, tuple(img_pts[0].ravel().astype(int)), tuple(img_pts[2].ravel().astype(int)), (0,255,0), 2, cv2.LINE_AA)
        cv2.line(img, tuple(img_pts[0].ravel().astype(int)), tuple(img_pts[3].ravel().astype(int)), (0,0,255), 2, cv2.LINE_AA)

        cv2.putText(img, str(idx), tuple(img_pts[0].ravel().astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0))


    return img
