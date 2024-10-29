import numpy as np
import cv2 
import matplotlib.pyplot as plt
from read_sig import get_id,get_id_median, equalSig


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
    


def get_contour_bits(img, cnt, bits):
    '''
    cnt -  the sub-pixel accuracy contours obtained using find_squares
    returns markers signature for particular contour
    '''

    # assert(bits>0 and round(np.sqrt(bits))==np.sqrt(bits))
    # pixel_len = np.sqrt(bits)

    # the corners of expected output image
    corners = np.float32([[0,0], [bits,0], [bits, bits], [0, bits]])
    print(cnt.shape)
    print(corners.shape)

    M = cv2.getPerspectiveTransform(cnt, corners)
    warped = cv2.warpPerspective(img, M, (bits, bits) , flags=cv2.INTER_LINEAR)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('thesis_transformed.png', warped)
    ret, binary = cv2.threshold(warped,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    print(binary.shape)
    cv2.imwrite('thesis_warpedthresh.png', binary)
    
#     # Calculate the marker bits
    sig_id = get_id_median(binary)
    # print(sig_id)
    # cv2.imwrite('out.png', ret_img)

    return sig_id


def create_tag_dict(marker_image, marker_res):
    '''
    marker_image  - the input frame or the warped one ??
    res_sig       - all possible signatures of marker
    res_world_loc - location of corner points corresponding to each signature 
    '''
    h, w, c = marker_image.shape
    img = cv2.cvtColor(marker_image, cv2.COLOR_BGR2GRAY)
    contour_points = np.float32([[0,0], [w,0], [w, h], [0, h]])

    # the coordinate of the world points indicate the size of the image
    world_points = np.float32([[0,0,0], [30,0,0], [30, 30, 0], [0, 30, 0]])
    dict_sig = []
    dict_world_loc= []
    # plt.imshow(marker_image)
    # plt.show()
    # print(contour_points)
    # sig = get_contour_bits(marker_image, contour_points, bits)
    # print('Before', sig)
    # The number 4 indicates four possible orientations of the marker
    for i in range(4):
        sig = get_contour_bits(marker_image, contour_points, marker_res)
        print(i, sig)
        dict_sig.append(sig)
        dict_world_loc.append(world_points)
        # print('After', dict_sig[i])
        marker_image = cv2.rotate(marker_image, cv2.ROTATE_90_CLOCKWISE)
        world_points = np.roll(world_points, 1, axis=0)

    return dict_sig, dict_world_loc

def detect_tag(img, dict_sig, allowedMisses=10):
    '''
    The function returns all the markers found in the image
    img - color image

    return:
    results - dictionary of all the markes found in the image
    '''
    result = {"tag_corner":[],"tag_index":[]}
    # Placeholder for collecting missed bits - m

    cands = find_squares(img)
    print("Length of candidate contours and dictionary: ",len(cands), len(dict_sig))
    for cant_num in range(len(cands)):
        # cnt = cands[i]
        # print("candidate: ",cant_num)
        # print("Candidate: ",cands[cant_num].shape)
        sig = get_contour_bits(img, cands[cant_num], 700)
        # print("Test sig: ",sig)
        for i in range(len(dict_sig)):
            # print("Dict:", dict_sig[j])
            for j in range(len(dict_sig[i])):
                m = equalSig(sig, dict_sig[i][j], allowedMisses)
                print(m)
                if (m <= allowedMisses):
                    print(cands[cant_num])
                    print('match')
                    result["tag_corner"].append(cands[cant_num])
                    result["tag_index"].append((i,j))
                # break
    return result


if __name__ == "__main__":
    # TEST INDIVIDUAL CASES

    # Load the marker image and create the tag dictionary
    marker = cv2.imread("astrotag.png")
    dict_sig, dict_world_loc = create_tag_dict(marker, 700)

    # Print the dictionary of signatures
    print(dict_sig)

    # Load the test image
    img = cv2.imread("frame255.png")

    # Find squares in the test image
    cands = find_squares(img)

    if cands:
        # Get the contour bits for the first candidate square
        sig_id = get_contour_bits(img, cands[0], 700)
        print("Signature ID:", sig_id)
    else:
        print("No squares detected in the image.")