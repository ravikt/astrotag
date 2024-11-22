import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def detect_corners(gray):
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    return dst

def find_centroids(dst):
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    return centroids

def refine_corners(gray, centroids):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    return corners

def draw_corners(image, corners):
    for i in range(1, len(corners)):
        cv2.circle(image, (int(corners[i, 0]), int(corners[i, 1])), 5, (0, 0, 255), -1)

def fit_polygon(image, corners):
    points = np.array(corners[1:], dtype=np.int32)
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    return points

def apply_douglas_peucker(points):
    epsilon = 0.02 * cv2.arcLength(points, True)
    approx = cv2.approxPolyDP(points, epsilon, True)
    return approx

def draw_approximated_polygon(image, approx):
    cv2.polylines(image, [approx], isClosed=True, color=(255, 0, 0), thickness=2)

def display_result(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Corners and Fitted Polygon')
    plt.show()

def main(image_path):
    image, gray = load_image(image_path)
    dst = detect_corners(gray)
    centroids = find_centroids(dst)
    corners = refine_corners(gray, centroids)
    draw_corners(image, corners)
    points = fit_polygon(image, corners)
    approx = apply_douglas_peucker(points)
    draw_approximated_polygon(image, approx)
    display_result(image)
    print(f'Number of corners detected: {len(approx)}')

if __name__ == "__main__":
    image_path = 'corners_output.png'
    main(image_path)
