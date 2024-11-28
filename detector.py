import numpy as np
import cv2 
import matplotlib.pyplot as plt
from read_sig import get_id,get_id_median, equalSig
from quadrilateral import find_squares

    
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
    cv2.imwrite('transformed.png', warped)
    ret, binary = cv2.threshold(warped,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    print(binary.shape)
    cv2.imwrite('warpedthresh.png', binary)
    
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