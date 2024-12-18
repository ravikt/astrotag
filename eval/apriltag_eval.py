import os
import cv2
from pupil_apriltags import Detector

# Directory setup
input_dir = 'apriltag_100cm'
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)
missed_detection = 0

# Initialize detector
at_detector = Detector(
    families="tagStandard41h12",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# Get all PNG images
image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
total_images = len(image_files)

for image_file in image_files:
    # Load and process image
    image_path = os.path.join(input_dir, image_file)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = at_detector.detect(gray)

    if not results:
        missed_detection += 1
        print(f"Missed detection in {image_file}")
    else:
        # Draw detections
        for r in results:
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))

            # Draw bounding box
            cv2.line(image, ptA, ptB, (0, 255, 0), 2)
            cv2.line(image, ptB, ptC, (0, 255, 0), 2)
            cv2.line(image, ptC, ptD, (0, 255, 0), 2)
            cv2.line(image, ptD, ptA, (0, 255, 0), 2)

    # Save result for all cases
    output_path = os.path.join(output_dir, image_file)
    cv2.imwrite(output_path, image)

print(f"Total images processed: {total_images}")
print(f"Total missed detections: {missed_detection}")