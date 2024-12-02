import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def calculate_centroid(p):

    centroid = ((p[0][0]+p[1][0]+p[2][0])/3, (p[0][1]+p[1][1]+p[2][1])/3) 
    return centroid


# def get_median_pixel_value(binary, cx, cy):
#     # Extract the 3x3 neighborhood around (cx, cy)
#     neighborhood = binary[cx-1:cx+2, cy-1:cy+2].flatten()
#     # Calculate and return the median value
#     return np.median(neighborhood)

# def get_id_median(binary):
#     nodes = []
#     signature = []

#     with open('keypoints.txt') as file:
#         for line in file:
#             x1, y1, x2, y2, x3, y3, cx, cy = line.split()
#             x = np.array([[int(x1), int(y1)], [int(x2), int(y2)], [int(x3), int(y3)]], np.int32)
#             x.reshape((-1, 1, 2))

#             # Get the median pixel value in the 8-point neighborhood around the centroid
#             median_value = get_median_pixel_value(binary, int(cx), int(cy))
#             if median_value >= 128:
#                 signature.append(1)
#             else:
#                 signature.append(0)

#     return signature


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
