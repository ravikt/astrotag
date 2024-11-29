import random
import numpy as np
import cv2
import os

def create_marker(num_markers, codeword, keypoints="triangle_list.txt"):
    """
    Create markers with triangles colored according to codeword bits
    Args:
        num_markers: Number of markers to generate
        codeword: Binary string of 1s and 0s (1=white, 0=black)
        keypoints: Text file containing triangle vertices
    """
    DIM = 700
    markers = []
    
    for i in range(num_markers):
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
        cv2.imwrite(f'testmarker_{i}.png', img)
        
    return markers

# Example usage:
codeword = "010010001110001000011011110110101001111111001000"  # 48-bit codeword

if __name__ == "__main__":
    markers = create_marker(1, codeword)
