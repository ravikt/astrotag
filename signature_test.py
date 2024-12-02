import numpy as np
import cv2 
import matplotlib.pyplot as plt
from read_sig import get_id

# Read the two test images
img1 = cv2.imread('sig_test/reference.png', cv2.IMREAD_GRAYSCALE)
# Create rotations of the image
img1_90 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
img1_180 = cv2.rotate(img1, cv2.ROTATE_180)
img1_270 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
# img2 = cv2.imread('sig_test/test.png', cv2.IMREAD_GRAYSCALE)


# Compute binary strings using get_id function
binary_str1 = get_id(img1)
binary_str1_90 = get_id(img1_90)
binary_str1_180 = get_id(img1_180)
binary_str1_270 = get_id(img1_270)


print("Binary string for image 1:", ''.join(map(str, binary_str1)))
print("Binary string for image 1 rotated 90 degrees:", ''.join(map(str, binary_str1_90)))
print("Binary string for image 1 rotated 180 degrees:", ''.join(map(str, binary_str1_180)))
print("Binary string for image 1 rotated 270 degrees:", ''.join(map(str, binary_str1_270)))
# print("Binary string for image 2:", ''.join(map(str, binary_str2)))

# test="100101-101101-011001-101110-110111-011111-000101-110000"
  #90d="001111-011010-001111-000111-001100-111011-111101-100001"
# ref ="100101101101011001101110110111011111000101110000