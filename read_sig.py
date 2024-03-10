import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../astrotag.png")#, cv2.IMREAD_UNCHANGED)
#img = cv2.imread("test.png", cv2.IMREAD_UNCHANGED)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #blur = cv2.GaussianBlur(gray,(7,7),0)
# #cv2.imwrite('blur.png', blur)
# ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


# The binarization doesn't result in exact bimodal distribution
# This given the binary signature output misleading
# print(ret)
#plt.hist(img.ravel(),256,[0,256])
#plt.show()


def calculate_centroid(p):

    centroid = ((p[0][0]+p[1][0]+p[2][0])/12, (p[0][1]+p[1][1]+p[2][1])/12) 
    return centroid

def get_id(binary):
    
    img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    #blur = cv2.GaussianBlur(gray,(7,7),0)
    #cv2.imwrite('blur.png', blur)
    # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    nodes = []
    signature = []
    for i in range(12):
        with open('../traingles/triangle_list{}.txt'.format(i+1)) as file :
            for line in file :
                x1, y1, x2, y2, x3, y3 = line.split()
                # nodes.append([[int(x1), int(y1)], [int(x2), int(y2)], [int(x3), int(y3)]])
                x = np.array([[int(x1), int(y1)], [int(x2), int(y2)], [int(x3), int(y3)]], np.int32)
                x.reshape((-1,1,2))

                # Uncomment the following to draw the mesh
                #cv2.polylines(img, [x], True, (0, 0, 0, 255), 2)
                
                # print(x[0,2])
                c = calculate_centroid(x)
                
                cv2.circle(img, (int(c[0]), int(c[1])), 4 , (255, 0, 0), cv2.FILLED)
                if (binary[int(c[0]), int(c[1])] > 128.0):
                    #print(c, (int(c[0]), int(c[1])), binary[int(c[0]), int(c[1])])
                    signature.append(1)
                    cv2.putText(img, '1',(int(c[0]), int(c[1])),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2, cv2.LINE_AA) 
                else:
                    signature.append(0)
                    cv2.putText(img, '0',(int(c[0]), int(c[1])),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2, cv2.LINE_AA)

    return img, signature

# im, sig = get_id(img)

# cv2.imwrite('out.png', im)
# print(signature)
# cv2.imwrite('out_gray.png', im2)
