import cv2
import numpy as np
# import mfd_descriptor
import json
from read_sig import get_keypoints
from fk_desc import fk_combined

import matplotlib.pyplot as plt

def visualize_keypoints_and_descriptors(image, keypoints, descriptors):
    # Convert image to RGB for plotting
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Plot the keypoints on the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    
    # Plot keypoints and their coordinates
    for kp in keypoints:
        x, y = kp[0], kp[1]
        plt.plot(x, y, 'ro')  # Red dot for keypoints
        # Add coordinate text with offset for visibility
        plt.text(x + 5, y + 5, f'({int(x)},{int(y)})', 
                color='yellow', fontsize=12, 
                bbox=dict(facecolor='black', alpha=0.7))
    
    plt.title('Keypoints with Coordinates')
    plt.axis('on')  # Show axes for reference
    plt.grid(True, alpha=0.3)  # Add light grid
    plt.show()
    
    # Display descriptor information
    print("\nKeypoint Descriptors:")
    for i, (kp, desc) in enumerate(zip(keypoints, descriptors)):
        print(f"Keypoint {i+1} at ({int(kp[0])},{int(kp[1])}): {desc}")

# Example usage

keypoints = get_keypoints('keypoints.txt')
# Read original image
image = cv2.imread('../marker/marker_0.png', cv2.IMREAD_GRAYSCALE)

# # Create rotated versions
# image_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
# image_180 = cv2.rotate(image, cv2.ROTATE_180)
# image_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

descriptors = fk_combined(image, keypoints)

visualize_keypoints_and_descriptors(image, keypoints, descriptors)
# visualize_keypoints_and_descriptors(image_90, keypoints, descriptors)
# visualize_keypoints_and_descriptors(image_180, keypoints, descriptors)
# visualize_keypoints_and_descriptors(image_270, keypoints, descriptors)

print("Descriptors:", descriptors.shape)

# Loop through different rotations
# for angle in [0, 90, 180, 270]:
#     # Rotate image
#     height, width = image.shape
#     center = (width/2, height/2)
#     rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    
#     # Rotate keypoints
#     rotated_keypoints = []
#     for kp in keypoints:
#         x, y = kp
#         # Apply same rotation to keypoints
#         new_x = rotation_matrix[0][0]*x + rotation_matrix[0][1]*y + rotation_matrix[0][2]
#         new_y = rotation_matrix[1][0]*x + rotation_matrix[1][1]*y + rotation_matrix[1][2]
#         rotated_keypoints.append([new_x, new_y])

#     descriptors = fk_combined(rotated_image, rotated_keypoints)
    
#     print(f"\nRotation: {angle} degrees")
#     visualize_keypoints_and_descriptors(rotated_image, rotated_keypoints, descriptors)
#     print("Descriptors shape:", descriptors.shape)

