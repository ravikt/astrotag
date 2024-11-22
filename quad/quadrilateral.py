import cv2
import numpy as np

def read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image")
    return image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def apply_adaptive_threshold(gradient_magnitude):
    binary = cv2.adaptiveThreshold(gradient_magnitude, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return binary

def find_contours(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def detect_lines(contour_img):
    lines = cv2.HoughLines(contour_img, 1, np.pi/180, threshold=50)
    return lines

def convert_lines_to_cartesian(lines):
    cartesian_lines = []
    for rho, theta in lines[:8, 0]:  # Consider up to 8 strongest lines
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
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

def detect_quadrilaterals(image_path):
    image = read_image(image_path)
    output = image.copy()
    gray = preprocess_image(image)
    gradient_magnitude = gray
    binary = apply_adaptive_threshold(gradient_magnitude)
    contours = find_contours(binary)
    detected_quadrilaterals = []

    for idx, contour in enumerate(contours):
        if cv2.contourArea(contour) < 100:
            continue
        contour_img = np.zeros_like(binary)
        cv2.drawContours(contour_img, [contour], -1, 255, 1)
        lines = detect_lines(contour_img)
        if lines is not None and len(lines) >= 4:
            cartesian_lines = convert_lines_to_cartesian(lines)
            parallel_pairs = find_parallel_pairs(cartesian_lines)
            if len(parallel_pairs) >= 2:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    rect = cv2.minAreaRect(approx)
                    width = rect[1][0]
                    height = rect[1][1]
                    aspect_ratio = max(width, height) / min(width, height)
                    if 0.5 < aspect_ratio < 5.0:
                        detected_quadrilaterals.append(approx)
                        cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)
    return detected_quadrilaterals, output

def draw_results(image_path, save_path=None):
    try:
        quads, result_image = detect_quadrilaterals(image_path)
        print(f"Found {len(quads)} quadrilaterals")
        if save_path:
            cv2.imwrite(save_path, result_image)
            print(f"Results saved to {save_path}")
        return result_image
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

if __name__ == "__main__":
    draw_results('test_marker.png', 'output.png')
