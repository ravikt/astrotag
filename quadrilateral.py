import cv2
import numpy as np
import os
from scipy import signal

def order_contour(cnt):
    # extract x,y coordinates for all the corner points
    c0_x, c0_y = (cnt[0].ravel())
    c1_x, c1_y = (cnt[1].ravel())
    c2_x, c2_y = (cnt[2].ravel())
    c3_x, c3_y = (cnt[3].ravel())

    # calculate the center of the contour
    cx = (c0_x + c1_x + c2_x + c3_x) / 4.0
    cy = (c0_y + c1_y + c2_y + c3_y) / 4.0

    # if the first corner is top left, swap the diagonal
    if (c0_x <= cx and c0_y <= cy):
        cnt[[1,3]] = cnt[[3,1]]

    else:
        cnt[[0,1]] = cnt[[1,0]]
        cnt[[2,3]] = cnt[[3,2]]
    
    return cnt

def read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image")
    return image

def preprocess_image(image):
        # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    # Write enhanced image to file
    cv2.imwrite('enhanced.png', enhanced)

    return enhanced

    

def apply_adaptive_threshold(gradient_magnitude):
    binary = cv2.adaptiveThreshold(gradient_magnitude, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary

def find_contours(binary):
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    filtered_contours = []
    
    if hierarchy is not None:
        hierarchy = hierarchy[0]  # Get the first element which contains the hierarchy info
        for contour, h in zip(contours, hierarchy):
            # h[2] is first child, h[3] is parent
            if h[2] != -1:  # If contour has children
                # Add all child contours
                child_idx = h[2]
                while child_idx != -1:
                    filtered_contours.append(contours[child_idx])
                    child_idx = hierarchy[child_idx][0]  # Get next sibling
            elif h[3] == -1:  # If contour has no parent (is outer)
                filtered_contours.append(contour)
    
    return filtered_contours

def detect_lines(contour_img):
    lines = cv2.HoughLinesP(
        contour_img,
        rho=1,
        theta=np.pi / 180,
        threshold=20,       # Lower threshold to detect more lines
        minLineLength=10,   # Minimum length of line to detect
        maxLineGap=5        # Maximum allowed gap between line segments
    )
    return lines

def convert_lines_to_cartesian(lines):
    cartesian_lines = []
    for line in lines[:8]:  # Consider up to 8 lines
        x1, y1, x2, y2 = line[0]
        cartesian_lines.append(((x1, y1), (x2, y2)))
    return cartesian_lines

def find_parallel_pairs(cartesian_lines):
    parallel_pairs = []
    for i in range(len(cartesian_lines)):
        for j in range(i + 1, len(cartesian_lines)):
            angle1 = np.arctan2(cartesian_lines[i][1][1] - cartesian_lines[i][0][1],
                                cartesian_lines[i][1][0] - cartesian_lines[i][0][0])
            angle2 = np.arctan2(cartesian_lines[j][1][1] - cartesian_lines[j][0][1],
                                cartesian_lines[j][1][0] - cartesian_lines[j][0][0])
            angle_diff = abs(angle1 - angle2)
            if angle_diff < 0.1 or abs(angle_diff - np.pi) < 0.1:
                parallel_pairs.append((i, j))
    return parallel_pairs

def detect_quadrilaterals(image):
    # image = read_image(image_path)
    output = image.copy()
    gray = preprocess_image(image)
    gradient_magnitude = gray
    binary = apply_adaptive_threshold(gradient_magnitude)
    cv2.imwrite('binary.png', binary)
    contours = find_contours(binary)
    
    # Criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    detected_quadrilaterals = []

    for idx, contour in enumerate(contours):

        if cv2.contourArea(contour) < 20:
            continue
        contour_img = np.zeros_like(binary)
        cv2.drawContours(contour_img, [contour], -1, 255, 1)
        lines = detect_lines(contour_img)
        
        if lines is not None and len(lines) >= 4:
            cartesian_lines = convert_lines_to_cartesian(lines)
            parallel_pairs = find_parallel_pairs(cartesian_lines)
            if len(parallel_pairs) >= 2:
                epsilon = 0.09 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    rect = cv2.minAreaRect(approx)
                    width = rect[1][0]
                    height = rect[1][1]
                    aspect_ratio = max(width, height) / min(width, height)
                    if 0.5 < aspect_ratio < 2.0:
                        corners = cv2.cornerSubPix(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), np.float32(approx), (3,3), (-1,-1),criteria)
                        cnt_corners = order_contour(corners)
                        # detected_quadrilaterals.append(corners)
                        detected_quadrilaterals.append(np.float32(approx))

                        # cv2.drawContours(output, [np.asarray(corners).reshape((4,2)).astype(int)], -1, (0, 255, 0), 2)
                        # cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)
                        # print("Normal corner:", approx)
                        # print("Subpixel corner:", corners)
    # return detected_quadrilaterals, output
    return detected_quadrilaterals

def draw_results(image_path, save_path=None):
    image = read_image(image_path)
    try:
        quads, result_image = detect_quadrilaterals(image)
        print(f"Found {len(quads)} quadrilaterals")
        if save_path:
            cv2.imwrite(save_path, result_image)
            print(f"Results saved to {save_path}")
        # Return whether quadrilaterals were found along with the image
        return result_image, len(quads) > 0
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None, False

# The main block should be modified to:
if __name__ == "__main__":
    input_folder = 'test_images'
    output_folder = 'results'
    successful_detections = 0
    total_images = 0

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            total_images += 1
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f'output_{filename}')
            _, success = draw_results(input_path, output_path)
            if success:
                successful_detections += 1
    
    print(f"\nSummary: Found quadrilaterals in {successful_detections} out of {total_images} images")

