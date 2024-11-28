import numpy as np
import cv2 
import matplotlib.pyplot as plt


def order_contour(cnt):
    """
    Orders the contour points of a quadrilateral in a specific manner.
    This function takes a contour with four points and reorders them based on their
    relative positions to the center of the contour. The reordering ensures that the
    points are arranged in a consistent manner, which can be useful for further
    processing or analysis.
    Parameters:
    cnt (numpy.ndarray): A contour with four points, where each point is represented
                         as a 2D coordinate (x, y).
    Returns:
    numpy.ndarray: The reordered contour points.
    """
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

# Function to check if a contour is a square
def is_square(cnt, min_size=200):
    """
    Determines if a given contour approximates a square shape.
    Args:
        cnt (numpy.ndarray): The contour to be checked, typically obtained from cv2.findContours.
        min_size (int, optional): The minimum area size of the contour to be considered a square. Defaults to 200.
    Returns:
        tuple: (bool, numpy.ndarray) True and the approximated contour if the contour approximates a square shape and meets the size criteria, False otherwise and None.
    """
    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    # Check if the contour has 4 vertices, is convex, has a suitable aspect ratio, and meets the size criteria
    if (len(approx) == 4 and cv2.isContourConvex(approx) and 
        0.6 <= float(cv2.boundingRect(approx)[2]) / cv2.boundingRect(approx)[3] <= 1.1 and 
        cv2.contourArea(cnt) > min_size):
        return True, approx
    return False, None

def draw_corners(image, corners):
    """
    Draws circles on the corners of the detected square.
    """
    cv2.circle(image, np.int_(corners[0].ravel()), 2, (255, 0, 0), 2)
    cv2.circle(image, np.int_(corners[1].ravel()), 2, (0, 255, 0), 2)
    cv2.circle(image, np.int_(corners[2].ravel()), 2, (0, 0, 255), 2)
    cv2.circle(image, np.int_(corners[3].ravel()), 2, (255, 128, 0), 2)

def find_squares(img):
    """
    Detects and returns the corners of square shapes in the given image.
    This function converts the input image to grayscale, applies adaptive thresholding 
    to obtain a binary image, inverts the binary image to detect black borders, and 
    finds contours in the inverted binary image. It then filters out the contours that 
    form squares, refines their corner points, and returns the corners of the detected 
    squares.
    Args:
        img (numpy.ndarray): The input image in which squares are to be detected.
    Returns:
        list: A list of numpy arrays, each containing the corner points of a detected square.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to get a binary image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert the binary image to detect black borders
    binary_inverted = cv2.bitwise_not(binary)

    # Find contours in the inverted binary image
    contours, _ = cv2.findContours(binary_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Placeholder for candidate squares
    cands = []

    # Loop through the contours and filter out the squares
    for cnt in contours:
        # Criteria for corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

        # Check if the contour is a square
        is_square_flag, approx = is_square(cnt)
        if is_square_flag:
            # Refine the corner points
            print(np.float32(approx).shape)
            corners = cv2.cornerSubPix(gray, np.float32(approx), (5, 5), (-1, -1), criteria)
            
            # Order the corners in a consistent manner
            cnt_corners = order_contour(corners)

            # Append the corners to the candidate list
            cands.append(cnt_corners)

            # Optionally draw the corners on the image for visualization
            # draw_corners(img, cnt_corners)
            # cv2.drawContours(img, [cnt_corners], -1, (0, 255, 0), 2)
            # cv2.imwrite('thesis_marker_contour.png', img)
            # print(cnt_corners.shape)
    
    return cands