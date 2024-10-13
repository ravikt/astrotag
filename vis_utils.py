import cv2
import json
import numpy as np
from relative_nav import solve_relative_pose, solve_relative_pose_lie


def draw_tag(img, res):

    '''
    The function drawContour takes the contour list as a python array not np array
    '''
    # print(len(res))
    # print(res["tag_index"])
    for i in range(len(res["tag_index"])):
    
        corner = res["tag_corner"][i]
        corner = np.asarray(corner).reshape((4,2)).astype(int)
        cv2.drawContours(img, [corner], -1, (255, 0, 0))
        org = (corner[0][0], corner[0][1])

        idx = res["tag_index"][i]
        # cv2.putText(img, str(idx), org, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0))

    return img

def project_axes(axis, rvec, tvec, camera_matrix, dist_coeffs):
    """Project the 3D axes into the image coordinate system."""
    return cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)

def draw_axes_on_image(img, img_pts, idx):
    """Draw the axes on the image."""
    cv2.line(img, tuple(img_pts[0].ravel().astype(int)), tuple(img_pts[1].ravel().astype(int)), (255, 0, 0), 2, cv2.LINE_AA)
    cv2.line(img, tuple(img_pts[0].ravel().astype(int)), tuple(img_pts[2].ravel().astype(int)), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(img, tuple(img_pts[0].ravel().astype(int)), tuple(img_pts[3].ravel().astype(int)), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, str(idx), tuple(img_pts[0].ravel().astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                color=(0, 255, 255), thickness=2, fontScale=0.7)

def drawAxesWithPose(img, res, dict_world_loc, use_lie_algebra=True):
    """Main function to draw axes based on pose estimation."""
    with open('camera_params/refraction_camera_intrinsic_astrotag.json', 'r') as camera_data:
        camera_params = json.load(camera_data)

    mtx = np.array(camera_params["camera_matrix"], dtype="double")
    dist = np.array(camera_params["distortion_coefficients"])

    axis = np.float32([[0, 0, 0], [25, 0, 0], [0, 25, 0], [0, 0, -25]])

    # Iterate through detected markers
    for i in range(len(res["tag_index"])):
        corner = res["tag_corner"][i]
        idx, rot_idx = res["tag_index"][i]

        corner = np.asarray(corner).reshape((4, 2)).astype('float32')

        # Choose the pose estimation method
        object_points = dict_world_loc[idx][rot_idx]  # 3D points corresponding to the detected marker

        if use_lie_algebra:
            pose_estimate = solve_relative_pose_lie(object_points, corner, mtx, dist)
            rvec = pose_estimate[:3]  # Extract rotation vector
            tvec = pose_estimate[3:]   # Extract translation vector
        else:
            rvec, tvec = solve_relative_pose(object_points, corner, mtx, dist)

        # Project the axes from the marker coordinate system to the image coordinate system
        img_pts, _ = project_axes(axis, rvec, tvec, mtx, dist)

        # Draw axes on the image
        draw_axes_on_image(img, img_pts, idx)

    return img