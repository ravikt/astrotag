import cv2
import numpy as np
import mfd_descriptor
import json
from read_sig import get_keypoints
from fk_desc import fk_combined



import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_keypoints_and_descriptors(image, keypoints, descriptors):
    # Convert image to RGB for plotting
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Plot the keypoints on the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    for kp in keypoints:
        plt.plot(kp[0], kp[1], 'ro')  # Red dot for keypoints
    
    plt.title('Keypoints on Image')
    plt.show()
    
    # Display the descriptors
    for i, desc in enumerate(descriptors):
        print(f"Descriptor for keypoint {keypoints[i]}: {desc}")

# Example usage

keypoints = get_keypoints('keypoints.txt')
image = cv2.imread('marker/lasrtag_dictionary/thesis_b_marker_0.png', cv2.IMREAD_GRAYSCALE)
# marker_path = 'marker/frame2.png'
# marker = cv2.imread(marker_path)
# keypoints = [(100, 100), (150, 150)]  # list of (x,y) coordinates
descriptors = fk_combined(image, keypoints)

visualize_keypoints_and_descriptors(image, keypoints, descriptors)
