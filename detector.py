import numpy as np
import cv2 
import matplotlib.pyplot as plt
from read_sig import get_id, hamming_distance, get_keypoints
from quadrilateral import detect_quadrilaterals
from custom_descriptor.fk_desc import fk_combined
# from grs.rs_codec import Generalized_Reed_Solomon
# from grs.message_generator import binary_string_to_int_list,int_list_to_binary_string
# from grs import Generalized_Reed_Solomon, binary_string_to_int_list, int_list_to_binary_string



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
    # print("Contour points:", cnt.shape)
    # print("Corner points:", corners.shape)

    M = cv2.getPerspectiveTransform(cnt, corners)
    # Debug prints
    
    warped = cv2.warpPerspective(img, M, (bits, bits) , flags=cv2.INTER_LINEAR)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('marker_transformed.png', warped)
    ret, binary = cv2.threshold(warped,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # print(binary.shape)
    cv2.imwrite('marker_warpedthresh.png', binary)
    
#     # Calculate the marker signature suing mask based method
    # sig_id = get_id(binary)

    # Calculate the marker signature using FK descriptor
    keypoints = get_keypoints('keypoints.txt')
    fk_descriptor = fk_combined(binary, keypoints)
    sig_id = ''.join(fk_descriptor.astype(str).flatten())
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



def detect_tag(img, dict_sig, allowedMisses=1000):
    '''
    The function returns all the markers found in the image
    img - color image

    return:
    results - dictionary of all the markes found in the image
    '''
    result = {"corner":[],"index":[], "world_loc":[]}
    # grs_encoder = Generalized_Reed_Solomon(2, 48, 24, 1, 1, None, False, False)
    # 0 = missed, 1 = detected
    count = 0
    # Placeholder for collecting missed bits - m
    cands = detect_quadrilaterals(img)
    # print("Length of candidate contours and dictionary: ",len(cands), len(dict_sig))
    for cant_num in range(len(cands)):
        # cnt = cands[i]
        # print("candidate: ",cant_num, cands[cant_num])
        sig = get_contour_bits(img, cands[cant_num], 700)
        encoded_bits = ''.join(map(str, sig))
        # print(encoded_bits)
        
        for idx, dictionary in enumerate(dict_sig):
            for orientation in ['0', '90', '180', '270']:
                message = dictionary[orientation]["signature"]
                m = hamming_distance(encoded_bits, message)
                # print("hamming distance:",m)
                if (m <= allowedMisses):
                    print('match')
                    result["corner"].append(cands[cant_num])
                    result["index"].append((idx))
                    result["world_loc"].append(dictionary[orientation]["world_points"])
    
    if len(result['index']) > 0 and 19 in result['index']:
        count=1
        
    return result, count
    # return(len(cands))


