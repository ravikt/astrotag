import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import os
# import descriptor_utils as du
import metrics as m
import plot_utils as pu




# Set the directory path where transformed images are saved
image_dir = "transformed_images"

# Descriptor list for comparison
descriptors = ["SIFT", "ORB", "MFD"]

# View current descriptors list in main.py
print("Current descriptors:", descriptors)

# Initialize a dictionary to hold evaluation metrics
metrics_data = {desc: {'accuracy': [], 'time': [], 'rotation': [], 'repeatability': {'rotation': []}} for desc in descriptors}


def main():
    # Load the original image
    image1_path = os.path.join(image_dir, "original.png")
    image1 = cv2.imread(image1_path, 0)
    
    if image1 is None:
        print("Error: Could not load the original image.")
        return

    # Evaluate descriptors for rotation robustness
    rotation_angles = [0, 15, 30, 45, 60, 90]
    for angle in rotation_angles:
        rotated_image_path = os.path.join(image_dir, f"rotated/rotated_{angle}deg.png")
        image_rotated = cv2.imread(rotated_image_path, 0)
        
        for descriptor in descriptors:
            # Add debug print in main loop
            print(f"Processing descriptor: {descriptor}")
            accuracy_rot, time_rot = m.evaluate_descriptor(descriptor, image1, image_rotated)
            metrics_data[descriptor]['accuracy'].append(accuracy_rot)
            metrics_data[descriptor]['time'].append(time_rot)
            metrics_data[descriptor]['rotation'].append(angle)

            repeatability_rot = m.compute_repeatability(descriptor, image1, image_rotated)
            metrics_data[descriptor]['repeatability']['rotation'].append(repeatability_rot)

    # Generate visualizations for each rotation angle
    for angle in rotation_angles:
        rotated_image_path = os.path.join(image_dir, f"rotated/rotated_{angle}deg.png")
        image_rotated = cv2.imread(rotated_image_path, cv2.IMREAD_GRAYSCALE)
        
        for descriptor in descriptors:
            output_path = f"matches/matches_{descriptor}_{angle}deg.png"
            m.visualize_matches(descriptor, image1, image_rotated, output_path)

    # Plot the results
    pu.plot_all_metrics(metrics_data, descriptors)

if __name__ == "__main__":
    main()
