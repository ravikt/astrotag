import numpy as np
from scipy.optimize import least_squares
from typing import Tuple, Optional
import cv2
import json

def solvePnP(objectPoints: np.ndarray, imagePoints: np.ndarray, cameraMatrix: np.ndarray, 
             distCoeffs: Optional[np.ndarray] = None, flags: int = 0) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Finds an object pose from 3D-2D point correspondences.
    
    Args:
    objectPoints: Array of object points in the object coordinate space, Nx3 1-channel or 1xN/Nx1 3-channel,
                  where N is the number of points. vector<Point3f> can be also passed here.
    imagePoints: Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
                 where N is the number of points. vector<Point2f> can be also passed here.
    cameraMatrix: Input camera matrix A = [[fx 0 cx],
                                           [0 fy cy],
                                           [0 0 1]].
    distCoeffs: Input vector of distortion coefficients (k1,k2,p1,p2,k3[,k4,k5,k6]) of 5 or 8 elements.
                If the vector is NULL/empty, the zero distortion coefficients are assumed.
    flags: Method for solving a PnP problem (currently ignored in this implementation).
    
    Returns:
    retval: bool indicating whether the solution was found.
    rvec: Output rotation vector that, together with tvec, brings points from the model coordinate system
          to the camera coordinate system.
    tvec: Output translation vector.
    """
    
    # Ensure correct shape of input arrays
    objectPoints = np.asarray(objectPoints).reshape(-1, 3)
    imagePoints = np.asarray(imagePoints).reshape(-1, 2)
    
    if distCoeffs is not None:
        distCoeffs = np.asarray(distCoeffs).flatten()
        if len(distCoeffs) >= 5:
            # Apply distortion correction
            k1, k2, p1, p2, k3 = distCoeffs[:5]
            x, y = imagePoints[:, 0], imagePoints[:, 1]
            r2 = x**2 + y**2
            r4 = r2**2
            r6 = r2**3
            
            # Radial distortion
            xdist = x * (1 + k1*r2 + k2*r4 + k3*r6)
            ydist = y * (1 + k1*r2 + k2*r4 + k3*r6)
            
            # Tangential distortion
            xdist = xdist + (2*p1*x*y + p2*(r2 + 2*x**2))
            ydist = ydist + (p1*(r2 + 2*y**2) + 2*p2*x*y)
            
            imagePoints = np.column_stack((xdist, ydist))
    
    # Ensure objectPoints and imagePoints have the same number of points
    n_points = min(objectPoints.shape[0], imagePoints.shape[0])
    objectPoints = objectPoints[:n_points]
    imagePoints = imagePoints[:n_points]
    
    # Normalize image points
    fx, fy = cameraMatrix[0, 0], cameraMatrix[1, 1]
    cx, cy = cameraMatrix[0, 2], cameraMatrix[1, 2]
    u = (imagePoints[:, 0] - cx) / fx
    v = (imagePoints[:, 1] - cy) / fy
    
    # Build matrices for estimation
    A1, A2 = build_matrices(objectPoints.T, u, v)
    K = calculate_k_matrix(A1, A2)
    C = build_c_matrix(K)
    
    # Estimate pose
    R_estimated, q = estimate_pose(C)
    t_estimated = calculate_translation(A1, A2, R_estimated)
    
    # Convert rotation matrix to rotation vector
    rvec = matrix_to_rodrigues(R_estimated)
    
    # Reshape rvec and tvec to match OpenCV's format
    rvec = rvec.reshape(3, 1)
    tvec = t_estimated.reshape(3, 1)
    
    # Print the estimated pose for debugging
    print(f"Estimated Rotation Vector (rvec): {rvec}")
    print(f"Estimated Translation Vector (tvec): {tvec}")
    
    # Return the results in the format of OpenCV's solvePnP
    return True, rvec, tvec

def matrix_to_rodrigues(R):
    """Convert a rotation matrix to a rotation vector."""
    theta = np.arccos((np.trace(R) - 1) / 2)
    if np.isclose(theta, 0):
        return np.zeros(3)
    else:
        r = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        return theta * r / (2 * np.sin(theta))

def build_matrices(world_points: np.ndarray, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build A1 and A2 matrices for pose estimation."""
    n = world_points.shape[1]
    A1 = np.zeros((2*n, 9))
    A2 = np.zeros((2*n, 3))
    
    for i in range(n):
        A1[2*i] = [world_points[0, i], 0, -u[i]*world_points[0, i],
                   world_points[1, i], 0, -u[i]*world_points[1, i],
                   world_points[2, i], 0, -u[i]*world_points[2, i]]
        A1[2*i+1] = [0, world_points[0, i], -v[i]*world_points[0, i],
                     0, world_points[1, i], -v[i]*world_points[1, i],
                     0, world_points[2, i], -v[i]*world_points[2, i]]
        A2[2*i:2*i+2] = [[1, 0, -u[i]],
                         [0, 1, -v[i]]]
    
    return A1, A2

def calculate_k_matrix(A1: np.ndarray, A2: np.ndarray) -> np.ndarray:
    """Calculate K matrix for pose estimation."""
    return A1 - A2 @ np.linalg.inv(A2.T @ A2) @ A2.T @ A1

def build_c_matrix(K: np.ndarray) -> np.ndarray:
    """Build C matrix for pose estimation."""
    n = K.shape[0] // 2
    C = np.zeros((2*n, 12))
    for i in range(2*n):
        C[i] = [K[i, 0] - K[i, 4] - K[i, 8],
                K[i, 4] - K[i, 0] - K[i, 8],
                K[i, 8] - K[i, 0] - K[i, 4],
                2*(K[i, 1] + K[i, 3]),
                2*(K[i, 2] + K[i, 6]),
                2*(K[i, 5] + K[i, 7]),
                2*(K[i, 7] - K[i, 5]),
                2*(K[i, 2] - K[i, 6]),
                2*(K[i, 3] - K[i, 1]),
                K[i, 0], K[i, 4], K[i, 8]]
    return C

def objective_function(q: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Objective function for least squares optimization."""
    Q = np.array([q[0]**2, q[1]**2, q[2]**2, q[0]*q[1], q[0]*q[2], q[1]*q[2], q[0], q[1], q[2], 1, 1, 1])
    return C @ Q

def estimate_pose(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate camera pose using least squares optimization."""
    result = least_squares(lambda q: objective_function(q, C), [0, 0, 0])
    q = result.x
    
    skew_q = np.array([[0, -q[2], q[1]],
                       [q[2], 0, -q[0]],
                       [-q[1], q[0], 0]])
    R_estimated = np.linalg.inv(np.eye(3) + skew_q) @ (np.eye(3) - skew_q)
    
    return R_estimated, q

def calculate_translation(A1: np.ndarray, A2: np.ndarray, R_estimated: np.ndarray) -> np.ndarray:
    """Calculate translation vector."""
    r_vec = R_estimated.flatten()
    return -np.linalg.inv(A2.T @ A2) @ A2.T @ A1 @ r_vec

def drawAxesWithPose(img, res, dict_world_loc):
    with open('camera_params/refraction_camera_intrinsic_astrotag.json', 'r') as camera_data:
        camera_params = json.load(camera_data)

    mtx = np.array(camera_params["camera_matrix"], dtype="double")
    dist = np.array(camera_params["distortion_coefficients"])

    axis = np.float32([[0, 0, 0], [25, 0, 0], [0, 25, 0], [0, 0, -25]])

    for i in range(len(res["tag_index"])):
        corner = res["tag_corner"][i]
        idx, rot_idx = res["tag_index"][i]

        corner = np.asarray(corner).reshape((4, 2)).astype('float32')

        # Find the rotation and translation vectors using custom solvePnP
        ret, rvecs, tvecs = cv2.solvePnP(dict_world_loc[idx][rot_idx], corner, mtx, dist)

        # Project points from marker coordinate system to image coordinate system
        img_pts, _ = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        # Print the projected points for debugging
        print(f"Projected Points (img_pts) for Tag Index {idx}: {img_pts}")

        # Ensure img_pts is in the correct format
        img_pts = img_pts.reshape(-1, 2).astype(int)

        # Check for infinities and replace them with a default value (e.g., 0)
        img_pts = np.where(np.isinf(img_pts), 0, img_pts)

        # Draw lines
        if img_pts.shape[0] == 4:
            cv2.line(img, tuple(img_pts[0]), tuple(img_pts[1]), (255, 0, 0), 2, cv2.LINE_AA)
            cv2.line(img, tuple(img_pts[0]), tuple(img_pts[2]), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(img, tuple(img_pts[0]), tuple(img_pts[3]), (0, 0, 255), 2, cv2.LINE_AA)

            # Add text
            cv2.putText(img, str(idx), tuple(img_pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 
                        color=(0, 255, 255), thickness=2, fontScale=0.7)
        else:
            print(f"Invalid projected points for Tag Index {idx}: {img_pts}")

    return img

# Example usage
if __name__ == "__main__":
    # Load your image and other necessary data
    img = cv2.imread('frame0.png')
    res = {
        "tag_index": [(0, 0)],  # Example indices
        "tag_corner": [
            [[241.99294, 153.13281],
             [296.4888,  154.90828],
             [286.17032, 207.77171],
             [233.32893, 208.17723]]  # Example corners for tag 0
        ]
    }
    dict_world_loc = {
        0: [[[ 0.,  0.,  0.],
             [30.,  0.,  0.],
             [30., 30.,  0.],
             [ 0., 30.,  0.]]]  # Example 3D points for tag 0
    }

    img_with_axes = drawAxesWithPose(img, res, dict_world_loc)
    cv2.imwrite('debug.png', img_with_axes)