import cv2
import json
import numpy as np

    # with open ('misses.txt', 'a') as file:
    #     file.write(f"{missed_number}\n")

def equalSig(sig1, sig2, allowedMisses=0):
    # global missed_number
    misses = 0
    # missed_number = 0
    for i in range(len(sig1)):
        
        if sig1[i] != sig2[i]:
            misses = misses + 1

    # print('misses:', misses)
    return misses#<=allowedMisses
    # return missed_number

def draw_tag(img, res):

    '''
    The function drawContour takes the contour list as a python array not np array
    '''
    print(len(res))
    print(res["tag_index"])
    for i in range(len(res["tag_index"])):
    
        corner = res["tag_corner"][i]
        corner = np.asarray(corner).reshape((4,2)).astype(int)
        cv2.drawContours(img, [corner], -1, (255, 0, 0))
        org = (corner[0][0], corner[0][1])

        idx = res["tag_index"][i]
        # cv2.putText(img, str(idx), org, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0))

    return img

def drawAxesWithPose(img, res, dict_world_loc):

 
    with open('camera_params/camera_intrinsic_chaser.json', 'r') as camera_data:
        camera_params = json.load(camera_data)

    mtx = np.array(camera_params["camera_matrix"], dtype="double")
    dist= np.array(camera_params["distortion_coefficients"])

    axis = np.float32([[0,0,0], [25,0,0], [0, 25, 0], [0, 0, -25]])

    # Convert the points from marker coordinate system to image coordinate system 
    print('length of tag index:', res["tag_index"])
    for i in range(len(res["tag_index"])):

        corner = res["tag_corner"][i]
        idx, rot_idx = res["tag_index"][i]

        corner = np.asarray(corner).reshape((4,2)).astype('float32')
        print(dict_world_loc[idx][rot_idx])
        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv2.solvePnP(dict_world_loc[idx][rot_idx], corner, mtx, dist)

        # draw axis on the marker
        # project points from marker coordinate system in the image coordinate system 

        img_pts, _ = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        # print(tuple(img_pts[0].ravel()))
        cv2.line(img, tuple(img_pts[0].ravel().astype(int)), tuple(img_pts[1].ravel().astype(int)), (255,0,0), 2, cv2.LINE_AA) 
        cv2.line(img, tuple(img_pts[0].ravel().astype(int)), tuple(img_pts[2].ravel().astype(int)), (0,255,0), 2, cv2.LINE_AA)
        cv2.line(img, tuple(img_pts[0].ravel().astype(int)), tuple(img_pts[3].ravel().astype(int)), (0,0,255), 2, cv2.LINE_AA)

        cv2.putText(img, str(idx), tuple(img_pts[0].ravel().astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 
                    color=(0, 255, 255), thickness=2, fontScale=0.7)


    return img
