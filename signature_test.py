import numpy as np
import cv2 
import matplotlib.pyplot as plt
from read_sig import get_id

# Read the two test images
img1 = cv2.imread('sig_test/reference.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('sig_test/test.png', cv2.IMREAD_GRAYSCALE)


# Compute binary strings using get_id function
binary_str1 = get_id(img1)
binary_str2 = get_id(img2)

print("Binary string for image 1:", ''.join(map(str, binary_str1)))
print("Binary string for image 2:", ''.join(map(str, binary_str2)))

# ref="000111111010100110011101101111101111100001010100"
# a  ="000111111010100110011101101111101111100001011100"
# b  ="000111111010100110011101101111101111100001011100"