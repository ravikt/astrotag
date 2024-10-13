import json
import numpy as np
import cv2
from scipy.optimize import least_squares

def solve_pose_with_opencv(object_points, image_points, camera_matrix, dist_coeffs):
    _, rvecs, tvecs = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    return rvecs, tvecs

def solve_relative_pose(object_points, image_points, camera_matrix, dist_coeffs=None, use_extrinsic_guess=False, initial_rvec=None, initial_tvec=None):
    """
    Solve for the relative pose (rotation and translation) of an object using 3D object points and their corresponding 2D image points.

    Parameters:
    ----------
    object_points : np.ndarray
        A Nx3 array of 3D points in the object coordinate space, where N is the number of points.
        
    image_points : np.ndarray
        A Nx2 array of 2D points in the image coordinate space corresponding to the object points.
        
    camera_matrix : np.ndarray
        A 3x3 intrinsic camera matrix.
        
    dist_coeffs : np.ndarray, optional
        A 1xN array of distortion coefficients (N is the number of distortion parameters). If None, no distortion is considered. Default is None.
        
    use_extrinsic_guess : bool, optional
        If True, use the provided initial_rvec and initial_tvec as the starting point for the optimization. Default is False.
        
    initial_rvec : np.ndarray, optional
        A 1x3 array representing the initial rotation vector. Required if use_extrinsic_guess is True.
        
    initial_tvec : np.ndarray, optional
        A 1x3 array representing the initial translation vector. Required if use_extrinsic_guess is True.

    Returns:
    -------
    rvec : np.ndarray
        A 1x3 array representing the estimated rotation vector.
        
    tvec : np.ndarray
        A 1x3 array representing the estimated translation vector.

    Example:
    --------
    object_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)  # 3D points
    image_points = np.array([[100, 150], [200, 150], [100, 250]], dtype=np.float32)  # Corresponding 2D points
    camera_matrix = np.array([[800, 0, 320],
                               [0, 800, 240],
                               [0, 0, 1]], dtype=np.float32)  # Example camera matrix
    rvec, tvec = solve_relative_pose(object_points, image_points, camera_matrix)
    """
    
    # Objective function for least squares
    def reprojection_error(params, object_points, image_points, camera_matrix, dist_coeffs):
        rvec = params[:3]
        tvec = params[3:]
        projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        return (projected_points.squeeze() - image_points).ravel()

    # Initial guess
    if use_extrinsic_guess and initial_rvec is not None and initial_tvec is not None:
        initial_guess = np.hstack((initial_rvec, initial_tvec))
    else:
        initial_guess = np.zeros(6)  # Defaults to zero if no guess provided

    result = least_squares(reprojection_error, initial_guess, args=(object_points, image_points, camera_matrix, dist_coeffs))

    rvec = result.x[:3]
    tvec = result.x[3:]
    return rvec, tvec


import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.linalg import expm, logm

def solve_relative_pose_lie(object_points, image_points, camera_matrix, dist_coeffs=None, use_extrinsic_guess=False, initial_pose=None):
    """
    Solve for the relative pose (rotation and translation) of an object using 3D object points 
    and their corresponding 2D image points, utilizing Lie algebra representation.

    Parameters:
    ----------
    object_points : np.ndarray
        A Nx3 array of 3D points in the object coordinate space, where N is the number of points.
        
    image_points : np.ndarray
        A Nx2 array of 2D points in the image coordinate space corresponding to the object points.
        
    camera_matrix : np.ndarray
        A 3x3 intrinsic camera matrix.
        
    dist_coeffs : np.ndarray, optional
        A 1xN array of distortion coefficients (N is the number of distortion parameters). 
        If None, no distortion is considered. Default is None.
        
    use_extrinsic_guess : bool, optional
        If True, use the provided initial_pose as the starting point for the optimization. 
        Default is False.
        
    initial_pose : np.ndarray, optional
        A 6x1 array representing the initial pose (3 for rotation and 3 for translation). 
        Required if use_extrinsic_guess is True.

    Returns:
    -------
    pose : np.ndarray
        A 6x1 array representing the estimated pose (rotation in the form of a vector and translation vector).

    Example:
    --------
    object_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)  # 3D points
    image_points = np.array([[100, 150], [200, 150], [100, 250]], dtype=np.float32)  # Corresponding 2D points
    camera_matrix = np.array([[800, 0, 320],
                               [0, 800, 240],
                               [0, 0, 1]], dtype=np.float32)  # Example camera matrix
    pose = solve_relative_pose_lie(object_points, image_points, camera_matrix)
    """

    # Objective function for least squares
    def reprojection_error(params, object_points, image_points, camera_matrix, dist_coeffs):
        # Convert params back to rotation matrix and translation vector
        theta = params[:3]  # Rotation vector
        tvec = params[3:]   # Translation vector
        R = expm(np.cross(np.eye(3), theta))  # Rotation matrix from the rotation vector
        pose_matrix = np.hstack((R, tvec.reshape(-1, 1)))
        projected_points, _ = cv2.projectPoints(object_points, pose_matrix[:, :3], pose_matrix[:, 3], camera_matrix, dist_coeffs)
        return (projected_points.squeeze() - image_points).ravel()

    # Initial guess
    if use_extrinsic_guess and initial_pose is not None:
        initial_guess = initial_pose
    else:
        initial_guess = np.zeros(6)  # Defaults to zero if no guess provided

    result = least_squares(reprojection_error, initial_guess, args=(object_points, image_points, camera_matrix, dist_coeffs))

    return result.x  # Return pose as a single vector (rotation + translation)

