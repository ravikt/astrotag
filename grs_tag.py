import random
import numpy as np
import cv2
import os
from read_sig import get_id
import json


def create_marker(idx, codeword, keypoints="keypoints.txt"):
    """
    Create markers with triangles colored according to codeword bits
    Args:
        num_markers: Number of markers to generate
        codeword: Binary string of 1s and 0s (1=white, 0=black)
        keypoints: Text file containing triangle vertices
    """
    DIM = 700
    markers = []
    
    # for i in range(num_markers):
    # Create white background
    img = np.zeros((DIM, DIM, 3), np.uint8)
    # img.fill(255)

    # Read triangle vertices and draw them
    triangle_idx = 0
    with open(keypoints) as file:
        for line in file:
            if triangle_idx >= len(codeword):
                break
                
            # Parse vertex coordinates
            coords = [float(x) for x in line.split()]
            triangle_pts = np.array([
                [int(coords[0]), int(coords[1])],
                [int(coords[2]), int(coords[3])], 
                [int(coords[4]), int(coords[5])]
            ], np.int32)
            
            # centroid = [int(coords[6]), int(coords[7])]

            # Set color based on codeword (1=white, 0=black)
            fill_color = (255,255,255) if codeword[triangle_idx] == '1' else (0,0,0)
            
            # Draw filled triangle
            cv2.fillPoly(img, [triangle_pts], fill_color)
            # Draw triangle outline
            # cv2.polylines(img, [triangle_pts], True, (1,1,1), thickness=0)
            
            # # Draw codeword bit and triangle index at centroid
            # text_color = (0,0,0) if codeword[triangle_idx] == '1' else (255,255,255)
            # text = f"{codeword[triangle_idx]}:{triangle_idx}"
            # cv2.putText(img, text, tuple(centroid), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            
            triangle_idx += 1
    
    sub_img = cv2.resize(img, (300, 300), interpolation= cv2.INTER_LINEAR)
    img[200:500, 200:500] = sub_img

    
    sub_sub_im = cv2.resize(sub_img, (125, 125), interpolation= cv2.INTER_LINEAR)
    img[287:412, 287:412] = sub_sub_im

    # Draw border the first border
    in_top_left = (200, 200)
    in_bottom_right = (500, 500)
    in_t = 20
    cv2.rectangle(img, (in_top_left[0]-in_t//2, in_top_left[1]-in_t//2),
                        (in_bottom_right[0]+in_t//2, in_bottom_right[1]+in_t//2), 
                        (255,255,255), in_t)
    
    # Draw border the second border
    ins_top_left = (287, 287)
    ins_bottom_right = (412, 412)
    ins_t = 10
    cv2.rectangle(img, (ins_top_left[0]-ins_t//2, ins_top_left[1]-ins_t//2),
                        (ins_bottom_right[0]+ins_t//2, ins_bottom_right[1]+ins_t//2), 
                        (255,255,255), ins_t)


    markers.append(img)
    
    # Save marker
    cv2.imwrite(f'marker/marker_{idx}.png', img)
        
    return markers


def debug_marker(img, encoded_msg):
    """Debug function to visualize triangle detection and color mapping"""
    # Add image inversion check
    print("\nChecking image properties:")
    print(f"Is image inverted? {np.mean(img) > 127}")
    
    debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    debug_mask = np.zeros_like(img)  # New debug mask
    triangle_idx = 0
    
    print("Image shape:", img.shape)
    print("Image dtype:", img.dtype)
    print("Image range:", np.min(img), "-", np.max(img))
    
    with open("keypoints.txt") as file:
        for line in file:
            if triangle_idx >= len(encoded_msg):
                break
                
            # Parse vertex coordinates
            coords = [float(x) for x in line.split()]
            triangle_pts = np.array([
                [int(coords[0]), int(coords[1])],
                [int(coords[2]), int(coords[3])], 
                [int(coords[4]), int(coords[5])]
            ], np.int32)
            
            # Create and visualize mask
            mask = np.zeros(img.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [triangle_pts], 255)
            
            # Get average pixel value in triangle region
            mean_val = cv2.mean(img, mask=mask)[0]
            
            # Draw triangle with color based on detection
            detected_bit = '1' if mean_val > 127 else '0'
            expected_bit = encoded_msg[triangle_idx]
            
            # Fix color assignment (green for match, red for mismatch)
            color = (0,255,0) if detected_bit == expected_bit else (0,0,255)
            
            # Debug output for each triangle
            print(f"Triangle {triangle_idx}:")
            print(f"  Mean value: {mean_val}")
            print(f"  Expected: {expected_bit}, Detected: {detected_bit}")
            
            # Draw triangle and text
            cv2.fillPoly(debug_mask, [triangle_pts], 255 if detected_bit == '1' else 0)
            cv2.polylines(debug_img, [triangle_pts], True, color, 2)
            cv2.putText(debug_img, f"{triangle_idx}:{detected_bit}({mean_val:.0f})", 
                       (triangle_pts[0][0], triangle_pts[0][1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            triangle_idx += 1
    
    # Save debug images
    cv2.imwrite('debug_marker.png', debug_img)
    cv2.imwrite('debug_mask.png', debug_mask)
    cv2.imwrite('original_gray.png', img)
    
    return debug_img


# def get_id(img):
#     """Decode marker ID from image"""
#     decoded_bits = []
    
#     with open("keypoints.txt") as file:
#         for line in file:
#             coords = [float(x) for x in line.split()]
#             triangle_pts = np.array([
#                 [int(coords[0]), int(coords[1])],
#                 [int(coords[2]), int(coords[3])], 
#                 [int(coords[4]), int(coords[5])]
#             ], np.int32)
            
#             # Create mask
#             mask = np.zeros(img.shape, dtype=np.uint8)
#             cv2.fillPoly(mask, [triangle_pts], 255)
            
#             # Get mean value and decode bit
#             mean_val = cv2.mean(img, mask=mask)[0]
#             # FIXED: Inverted logic - high mean value (white) should be '1'
#             bit = '1' if mean_val > 127 else '0'  # Changed from '0' if mean_val > 127 else '1'
#             decoded_bits.append(bit)
            
#     return decoded_bits


if __name__ == "__main__":
    # encoded_msg = "100101101101011001101110110111011111000101110000"
    
    # # Create and save marker
    # markers = create_marker(1, encoded_msg)
    
    # # Read and decode
    # img = cv2.imread('marker.png', cv2.IMREAD_GRAYSCALE)
    # decoded_msg = get_id(img)
    # decoded_str = ''.join(map(str, decoded_msg))
    
    # print("Original msg:", encoded_msg)
    # print("Decoded msg: ", decoded_str)
    
    # # Debug visualization
    # debug_img = debug_marker(img, encoded_msg)
    
    # # Print detailed comparison
    # print("\nBit-by-bit comparison:")
    # for i, (orig, dec) in enumerate(zip(encoded_msg, decoded_str)):
    #     if orig != dec:
    #         print(f"Position {i}: Expected {orig}, Got {dec}")
    
    # hamming_distance = sum(c1 != c2 for c1, c2 in zip(encoded_msg, decoded_str))
    # print(f"\nHamming distance: {hamming_distance}")

    with open('grs/output/message_pairs24.json', 'r') as file:
        params = json.load(file)
        for key, value in params.items():
            # print(key,value["message"])
            markers = create_marker(int(key)-1, value["codeword_grs"])
            print(value["codeword_grs"])