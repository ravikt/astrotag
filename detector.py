import numpy as np
import cv2 
import matplotlib.pyplot as plt
from img_utils import draw_grid, equalSig, drawArucos, drawAxesWithPose
from read_sig import get_id


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

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11, 7)

    cv2.imwrite('thesis_adaptivethresh.png', thresh)
    # plt.show()

    # findContour fins whiite blobs over black background, therefore following steps performs 
    # the binary inversion 
    thresh = cv2.bitwise_not(thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
    cv2.imwrite('thesis_contour.png',cv2.drawContours(thresh, contours, -1, (0, 0, 255)))
    # plt.imshow(thresh)
    cands = []
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.09*perimeter, True)
        x,y,w,h = cv2.boundingRect(c)
        aspect_ratio = float(w)/h
        if (len(approx) == 4 and cv2.contourArea(c)>200 and aspect_ratio<=1.0):
            print("Contour is convex")

            # area = cv2.contourArea(c)
            # hull = cv2.convexHull(c)
            # hull_area = cv2.contourArea(hull)
            # solidity = float(area)/hull_area

            


            print(aspect_ratio)
            # calculates the refined corner locations
            corners = cv2.cornerSubPix(img, np.float32(approx), (5,5), (-1,-1),criteria)
            
            # print(corners)
            cnt_corners = order_contour(corners)
            # print(np.int_(corners))
            # cands.append(np.int_(corners))
            cands.append(corners)

            cv2.circle(thresh, np.int_(corners[0].ravel()), 2, (255,0,0), 2)
            cv2.circle(thresh, np.int_(corners[1].ravel()), 2, (0,255,0), 2)
            cv2.circle(thresh, np.int_(corners[2].ravel()), 2, (0,0,255), 2)
            cv2.circle(thresh, np.int_(corners[3].ravel()), 2, (255,128,0), 2)

    # thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
    # print(len(cands))
    cv2.imwrite('thesis_thresh.png',thresh)
    return thresh, cands



def get_contour_bits(img, cnt, bits):
    '''
    cnt -  the sub-pixel accuracy contours obtained using find_squares
    returns markers signature for particular contour
    '''

    # assert(bits>0 and round(np.sqrt(bits))==np.sqrt(bits))
    # pixel_len = np.sqrt(bits)

    # the corners of expected output image
    corners = np.float32([[0,0], [bits,0], [bits, bits], [0, bits]])
    # print(cnt.shape)
    # print(corners.shape)

    M = cv2.getPerspectiveTransform(cnt, corners)
    warped = cv2.warpPerspective(img, M, (bits, bits) , flags=cv2.INTER_LINEAR)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('thesis_transformed.png', warped)
    ret, binary = cv2.threshold(warped,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    print(binary.shape)
    cv2.imwrite('thesis_warpedthresh.png', binary)
    # M = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # binary = cv2.erode(binary, M)
    # cv2.imwrite('thesis_eroded.png', binary)
    # grid_binary = cv2.cvtColor(binary,cv2.COLOR_GRAY2BGR)
    # cv2.imwrite('thesis_grid.png', grid_binary)
    # out = draw_grid(grid_binary, pixel_len, pixel_len )
    # cv2.imwrite('thesis_grid.png', out)

    
#     # Calculate the marker bits
    ret_img, sig_id = get_id(binary)
    print(sig_id)
    cv2.imwrite('out.png', ret_img)

#     for r in range(int(pixel_len)):
#         for c in range(int(pixel_len)):
#             x = r*pixel_len + (pixel_len/2)
#             y = c*pixel_len + (pixel_len/2)
            
#             if(binary[int(x),int(y)] >= 128):
#                 res.append(1)
#                 cv2.putText(grid_binary, '1',(int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 1, cv2.LINE_AA)

#             else:
#                 res.append(0)
#                 cv2.putText(grid_binary, '0',(int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 1, cv2.LINE_AA)


#         cv2.imwrite('diagnose{}.png'.format(r), grid_binary)
#     # print(res)
    
#     return res


# def loadMarkerDictionary(marker_image, marker_bits):
#     '''
#     marker_image  - the input frame or the warped one ??
#     res_sig       - all possible signatures of marker
#     res_world_loc - location of corner points corresponding to each signature 
#     '''
#     h, w, c = marker_image.shape
#     img = cv2.cvtColor(marker_image, cv2.COLOR_BGR2GRAY)
#     contour_points = np.float32([[0,0], [w,0], [w, h], [0, h]])

#     # the coordinate of the world points indicate the size of the image
#     world_points = np.float32([[0,0,0], [30,0,0], [30, 30, 0], [0, 30, 0]])
#     bits = marker_bits
#     dict_sig = []
#     dict_world_loc= []
#     # plt.imshow(marker_image)
#     # plt.show()
#     # print(contour_points)
#     # sig = get_contour_bits(marker_image, contour_points, bits)
#     # print('Before', sig)
#     # The number 4 indicates four possible orientations of the marker
#     for i in range(1):
#         sig = get_contour_bits(marker_image, contour_points, bits)
#         print(i, sig)
#         dict_sig.append(sig)
#         dict_world_loc.append(world_points)
#         # print('After', dict_sig[i])
#         marker_image = cv2.rotate(marker_image, cv2.ROTATE_90_CLOCKWISE)
#         world_points = np.roll(world_points, 1, axis=0)

#     return dict_sig, dict_world_loc

# def detectAruco(img, dict_sig, dict_world_loc, allowedMisses=2):
#     '''
#     THe function returns all the markers found in the image
#     img - color image

#     return:
#     results - dictionary of all the markes found in the image
#     '''
#     result = {"ar_corners":[],"ar_index":[]}

#     thresh, cands = find_squares(img)
#     print("Length of candidate contours and dictionary: ",len(cands), len(dict_sig))
#     for i in range(len(cands)):
#         cnt = cands[i]
#         print("candidate: ",i)
#         sig = get_contour_bits(img, cnt, 49)
#         # print("Test sig: ",sig)
#         for j in range(len(dict_sig)):
#             # print("Dict:", dict_sig[j])
#             if equalSig(sig, dict_sig[j], allowedMisses):
#                 print('match')
#                 result["ar_corners"].append(cnt)
#                 result["ar_index"].append(j)
#                 # break

#     return result


# TEST INDIVIDUAL CASES

# marker = cv2.imread("marker_aruco.png")
# dict_sig, dict_world_loc = loadMarkerDictionary(marker, 49)

# print(dict_sig)
# for i in range(500):
img = cv2.imread("../frame{}.png".format(0))
thresh, cands = find_squares(img)

# print(cands[0])
ret_im, sig_id = get_contour_bits(img, cands[0], 200)