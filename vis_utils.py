import cv2
import json
import numpy as np


def draw_tag(img, res):

    """
    Draw detected tags on an image.

    Args:
    img (numpy.ndarray): The input image on which to draw.
    res (dict): A dictionary containing detection results with keys:
        - 'tag_index': List of tag indices.
        - 'tag_corner': List of tag corner coordinates.

    Returns:
    numpy.ndarray: The image with drawn tags.

    Note:
    The function uses cv2.drawContours which expects contours as Python lists, not numpy arrays.
    """
    print(len(res))
    print(res["index"])
    for i in range(len(res["index"])):
    
        corner = res["corner"][i]
        corner = np.asarray(corner).reshape((4,2)).astype(int)
        cv2.drawContours(img, [corner], -1, (0, 255, 0))
        org = (corner[0][0], corner[0][1])

        idx = res["index"][i]
        cv2.putText(img, str(idx), org, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))

    return img

def drawAxesWithPose(img, res):
    
    with open('camera_utils/camera_intrinsic.json', 'r') as camera_data:
        camera_params = json.load(camera_data)

    mtx = np.array(camera_params["camera_matrix"], dtype="double")
    dist= np.array(camera_params["distortion_coefficients"])

    axis = np.float32([[0,0,0], [25,0,0], [0, 25, 0], [0, 0, -25]])

    # Convert the points from marker coordinate system to image coordinate system 
    print('length of tag index:', len(res["index"]))
    for i in range(len(res["index"])):

        corner = res["corner"][i]
        idx = res["index"][i]
        world_loc = np.float32(res["world_loc"][i])

        corner = np.asarray(corner).reshape((4,2)).astype('float32')
        # print(dict_world_loc[idx][rot_idx])
        # Find the rotation and translation vectors.
        # ret, rvecs, tvecs = cv2.solvePnP(dict_world_loc[idx][rot_idx], corner, mtx, dist)
        ret, rvecs, tvecs = cv2.solvePnP(world_loc, corner, mtx, dist)


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

def drawPoseCube(img, res):
    with open('camera_utils/camera_intrinsic.json', 'r') as camera_data:
        camera_params = json.load(camera_data)

    mtx = np.array(camera_params["camera_matrix"], dtype="double")
    dist = np.array(camera_params["distortion_coefficients"])

    # Define cube points (assuming cube size of 25 units)
    cube_points = np.float32([[0,0,0], [0,30,0], [30,30,0], [30,0,0],
                             [0,0,-30], [0,30,-30], [30,30,-30], [30,0,-30]])

    # Convert the points from marker coordinate system to image coordinate system 
    print('length of tag index:', len(res["index"]))
    for i in range(len(res["index"])):
        corner = res["corner"][i]
        idx = res["index"][i]
        world_loc = np.float32(res["world_loc"][i])

        corner = np.asarray(corner).reshape((4,2)).astype('float32')
        ret, rvecs, tvecs = cv2.solvePnP(world_loc, corner, mtx, dist)

        # Project cube points to image plane
        img_pts, _ = cv2.projectPoints(cube_points, rvecs, tvecs, mtx, dist)
        img_pts = img_pts.astype(int)

        # Draw bottom square
        for j in range(4):
            cv2.line(img, tuple(img_pts[j][0]), tuple(img_pts[(j+1)%4][0]), (0,255,0), 2)
        # Draw top square
        for j in range(4):
            cv2.line(img, tuple(img_pts[j+4][0]), tuple(img_pts[(j+1)%4+4][0]), (0,255,0), 2)
        # Draw vertical lines
        for j in range(4):
            cv2.line(img, tuple(img_pts[j][0]), tuple(img_pts[j+4][0]), (0,255,0), 2)

        cv2.putText(img, str(idx), tuple(img_pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 
                    color=(0, 255, 255), thickness=2, fontScale=0.7)

    return img
