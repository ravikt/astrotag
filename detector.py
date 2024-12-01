import numpy as np
import cv2 
import matplotlib.pyplot as plt
from read_sig import get_id, equalSig
# from grs.rs_codec import Generalized_Reed_Solomon
# from grs.message_generator import binary_string_to_int_list,int_list_to_binary_string
from grs import Generalized_Reed_Solomon, binary_string_to_int_list, int_list_to_binary_string

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


def find_squares(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,151, 31)

    cv2.imwrite('adaptivethresh.png', thresh)
    # plt.show()

    # findContour finds white blobs over black background, therefore following steps performs 
    # the binary inversion 
    thresh = cv2.bitwise_not(thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #
    print(len(contours))
    # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    rgb_im_contour = img.copy()
    cv2.imwrite('marker_contour.png',cv2.drawContours(rgb_im_contour, contours, -1, (0, 0, 255)))
    # plt.imshow(thresh)
    cands = []
    min_area = 500

    rgb_im_corner = img.copy()
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.09*perimeter, True)
        x,y,w,h = cv2.boundingRect(c)
        aspect_ratio = float(w)/h
        # print('Aspect Ratio:', aspect_ratio)
        if (len(approx) == 4) and cv2.contourArea(c)>min_area and 0.9<=aspect_ratio<=1.1: #
            # print("Contour is convex")

            # print(aspect_ratio)
            # calculates the refined corner locations
            corners = cv2.cornerSubPix(gray, np.float32(approx), (5,5), (-1,-1),criteria)
            
            # print(corners)
            cnt_corners = order_contour(corners)
            # print(np.int_(corners))
            # cands.append(np.int_(corners))
            cands.append(corners)

            cv2.circle(rgb_im_corner, np.int_(corners[0].ravel()), 2, (255,0,0), 2)
            cv2.circle(rgb_im_corner, np.int_(corners[1].ravel()), 2, (0,255,0), 2)
            cv2.circle(rgb_im_corner, np.int_(corners[2].ravel()), 2, (0,0,255), 2)
            cv2.circle(rgb_im_corner, np.int_(corners[3].ravel()), 2, (255,128,0), 2)

    # thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
    # print(len(cands))
    cv2.imwrite('marker_thresh.png',rgb_im_corner)
    return rgb_im_corner, cands



def get_contour_bits(img, cnt, bits):
    '''
    cnt -  the sub-pixel accuracy contours obtained using find_squares
    returns markers signature for particular contour
    '''

    # assert(bits>0 and round(np.sqrt(bits))==np.sqrt(bits))
    # pixel_len = np.sqrt(bits)

    # the corners of expected output image
    corners = np.float32([[0,0], [bits,0], [bits, bits], [0, bits]])
    # print(cnt)
    # print(corners.shape)

    M = cv2.getPerspectiveTransform(cnt, corners)
    warped = cv2.warpPerspective(img, M, (bits, bits) , flags=cv2.INTER_LINEAR)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('marker_transformed.png', warped)
    ret, binary = cv2.threshold(warped,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    print(binary.shape)
    cv2.imwrite('marker_warpedthresh.png', binary)
    
#     # Calculate the marker bits
    sig_id = get_id(binary)
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
        # print(i, sig)
        dict_sig.append(sig)
        dict_world_loc.append(world_points)
        # print('After', dict_sig[i])
        marker_image = cv2.rotate(marker_image, cv2.ROTATE_90_CLOCKWISE)
        world_points = np.roll(world_points, 1, axis=0)

    return dict_sig, dict_world_loc

# def detect_tag(img, dict_sig, allowedMisses=30):
#     '''
#     The function returns all the markers found in the image
#     img - color image

#     return:
#     results - dictionary of all the markes found in the image
#     '''
#     result = {"tag_corner":[],"tag_index":[]}
#     # Placeholder for collecting missed bits - m
#     count = 1
#     thresh, cands = find_squares(img)
#     # print("Length of candidate contours and dictionary: ",len(cands), len(dict_sig))
#     for cant_num in range(len(cands)):
#         # cnt = cands[i]
#         # print("candidate: ",cant_num)
#         sig = get_contour_bits(img, cands[cant_num], 700)
#         # print("Test sig: ",sig)
#         for i in range(len(dict_sig)):
#             # print("Dict:", dict_sig[j])
#             for j in range(len(dict_sig[i])):
#                 m = equalSig(sig, dict_sig[i][j], allowedMisses)
#                 # print(m)
#                 # if equalSig(sig, dict_sig[i][j], allowedMisses):
#                 if (m <= allowedMisses):
#                     print('match')
#                     result["tag_corner"].append(cands[cant_num])
#                     result["tag_index"].append((i,j))
    
#     if len(result['tag_index'])==0:
#         count=0
#     return result, count

def detect_tag(img, dict_sig):
    '''
    The function returns all the markers found in the image
    img - color image

    return:
    results - dictionary of all the markes found in the image
    '''
    result = {"tag_corner":[],"tag_index":[]}
    grs_encoder = Generalized_Reed_Solomon(2, 48, 24, 1, 1, None, False, False)

    # Placeholder for collecting missed bits - m
    thresh, cands = find_squares(img)
    print("Length of candidate contours and dictionary: ",len(cands), len(dict_sig))
    for cant_num in range(len(cands)):
        # cnt = cands[i]
        # print("candidate: ",cant_num)
        sig = get_contour_bits(img, cands[cant_num], 700)
        encoded_grs_bits = ''.join(map(str, sig))
        print(encoded_grs_bits)
        
        encoded_grs_int_list = binary_string_to_int_list(encoded_grs_bits)
        decoded_grs = grs_encoder.decode(encoded_grs_int_list)
        decoded_grs_bits = int_list_to_binary_string(decoded_grs)
        # print("Test sig: ",sig)
        for i in range(len(dict_sig)):
             
            message = dict_sig[i]
            # print(type(message))
        # print(f"Decoded GRS Message: {decoded_grs_bits[:message_length_bits]}")
            if decoded_grs_bits[:24] == str(message):
                # print(f"Verification failed for GRS codeword at ID {key}")
            
                print('match')
                result["tag_corner"].append(cands[cant_num])
                result["tag_index"].append((i))
    
    
    return result


if __name__=="__main__":
# TEST INDIVIDUAL CASES

# marker = cv2.imread("astrotag.png")
# dict_sig, dict_world_loc = create_tag_dict(marker, 700)

# print(dict_sig)
# for i in range(500):
    img = cv2.imread("test_images/frame{}_100cm.png".format(0))
    thresh, cands = find_squares(img)

    # print(cands[0])
    #sig_id = get_contour_bits(img, cands[0], 700)
