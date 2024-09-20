import cv2
import json
import numpy as np

def hamming_distance(seq1, seq2):
    """
    Calculate the Hamming distance between two sequences.
    
    Args:
    seq1, seq2: Sequences to compare (strings, lists, or any iterable)
    
    Returns:
    int: The Hamming distance
    
    Raises:
    ValueError: If sequences are of unequal length
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length")
    
    return sum(el1 != el2 for el1, el2 in zip(seq1, seq2))

def are_sequences_similar(seq1, seq2, max_misses=0):
    """
    Check if two sequences are similar within a specified tolerance.
    
    Args:
    seq1, seq2: Sequences to compare
    max_misses: Maximum allowed differences (default 0)
    
    Returns:
    bool: True if sequences are similar within the tolerance, False otherwise
    """
    return hamming_distance(seq1, seq2) <= max_misses

# Example usage:
# distance = hamming_distance("1011101", "1001001")
# is_similar = are_sequences_similar("1011101", "1001001", max_misses=2)

# def equalSig(sig1, sig2, allowedMisses=0):
#     # global missed_number
#     misses = 0
#     # missed_number = 0
#     for i in range(len(sig1)):
        
#         if sig1[i] != sig2[i]:
#             misses = misses + 1

#     # print('misses:', misses)
#     return misses#<=allowedMisses
#     # return missed_number

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
    
    with open('camera_utils/camera_intrinsic.json', 'r') as camera_data:
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
