import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_keypoints(filename, scale=1.0):
    keypoints = []

    with open(filename) as file:
        for line in file:
            x1, y1, x2, y2, x3, y3, cx, cy = line.split()
            # Convert string coordinates to float and scale them
            keypoints.append((float(x1) * scale, float(y1) * scale))
            keypoints.append((float(x2) * scale, float(y2) * scale))
            keypoints.append((float(x3) * scale, float(y3) * scale))

    return keypoints

def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Strings must be of equal length")
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def equalSig(sig1, sig2):
    """Calculate Hamming distance between two binary strings"""
    if len(sig1) != len(sig2):
        raise ValueError("Strings must be of equal length")
    
    # Calculate Hamming distance (number of positions at which strings differ)
    hamming_distance = sum(c1 != c2 for c1, c2 in zip(sig1, sig2))
    return hamming_distance

def calculate_centroid(p):

    centroid = ((p[0][0]+p[1][0]+p[2][0])/3, (p[0][1]+p[1][1]+p[2][1])/3) 
    return centroid


def get_id(img):
    """Decode marker ID from image"""
    decoded_bits = []
    
    with open("keypoints.txt") as file:
        for line in file:
            coords = [float(x) for x in line.split()]
            triangle_pts = np.array([
                [int(coords[0]), int(coords[1])],
                [int(coords[2]), int(coords[3])], 
                [int(coords[4]), int(coords[5])]
            ], np.int32)
            
            # Create mask
            mask = np.zeros(img.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [triangle_pts], 255)
            
            # Get mean value and decode bit
            mean_val = cv2.mean(img, mask=mask)[0]
            # FIXED: Inverted logic - high mean value (white) should be '1'
            bit = '1' if mean_val > 127 else '0'  # Changed from '0' if mean_val > 127 else '1'
            decoded_bits.append(bit)
            
    return decoded_bits
