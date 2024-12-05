import os
import cv2
import numpy as np

# Setup directories
input_dir = 'aruco_100cm'
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

# Initialize detector
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

# Get all PNG images
image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
missed_detection = 0
total_images = len(image_files)

for image_file in image_files:
    # Load image
    image_path = os.path.join(input_dir, image_file)
    image = cv2.imread(image_path)
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(image)
    
    if len(corners) == 0:
        missed_detection += 1
        print(f"Missed detection in {image_file}")
    else:
        # Process detected markers
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            
            # Convert coordinates to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            
            # Draw marker bounds
            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
    
    # Save result
    output_path = os.path.join(output_dir, image_file)
    cv2.imwrite(output_path, image)

print(f"Total images processed: {total_images}")
print(f"Total missed detections: {missed_detection}")
