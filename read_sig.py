import cv2
import numpy as np
import matplotlib.pyplot as plt

def equalSig(sig1, sig2, allowedMisses=0):
    # global missed_number
    misses = 0
    # missed_number = 0
    for i in range(len(sig1)):
        
        if sig1[i] != sig2[i]:
            misses = misses + 1

    # print('misses:', misses)
    return misses#<=allowedMisses
    # return missed_number

def calculate_centroid(p):

    centroid = ((p[0][0]+p[1][0]+p[2][0])/3, (p[0][1]+p[1][1]+p[2][1])/3) 
    return centroid

def get_id(binary):
    
    # img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    #blur = cv2.GaussianBlur(gray,(7,7),0)
    #cv2.imwrite('blur.png', blur)
    # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    nodes = []
    signature = []

    with open('astrotag/triangle_list.txt') as file :
        for line in file :
            x1, y1, x2, y2, x3, y3, cx, cy = line.split()
            # nodes.append([[int(x1), int(y1)], [int(x2), int(y2)], [int(x3), int(y3)]])
            x = np.array([[int(x1), int(y1)], [int(x2), int(y2)], [int(x3), int(y3)]], np.int32)
            x.reshape((-1,1,2))
            
            # print(x[0,2])
            # c = calculate_centroid(x)
            # cv2.putText(img, 'OR',(0,0),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2, cv2.LINE_AA) 

            # cv2.circle(img, (int(cx), int(cy)), 4 , (0, 255, 0), cv2.FILLED)
            if (binary[int(cx), int(cy)] >= 128):
                #print(c, (int(c[0]), int(c[1])), binary[int(c[0]), int(c[1])])
                signature.append(1)
                # cv2.putText(img, '1',(int(cx), int(cy)),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2, cv2.LINE_AA) 
            else:
                signature.append(0)
                # cv2.putText(img, '0',(int(cx), int(cy)),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2, cv2.LINE_AA)
            
            # Uncomment the following to draw the mesh
            # cv2.imwrite('results/frame.png',cv2.polylines(binary, [x], True, (35, 150, 200), 2))

    return signature


if __name__ == "__main__":
    sig = get_id(binary)

    cv2.imwrite('out.png', img)
    # print(sig)