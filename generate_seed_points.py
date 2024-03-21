
# Author: Ravi Kumar Thakur
# This script generates seed points for traingulation of marker template

import numpy as np 
import cv2
import random
import math
from mesh import point_on_triangle, trisample, solid_triangle, perimeter_points

random.seed(312345)
##### Draw Semi-Spidron
# create white image
dim = 700
img = np.zeros((dim,dim,3), np.uint8)
img.fill(255)


# Used for coorindtaes of template
triangles     = np.ones((12,3,2)) 
# Used for coordinates of triangles for encoding
sub_triangles = np.ones((12,3,2)) 

with open('astrotag_matrix.txt') as file:
    i = 0
    for line in file:
        x = line.split()
        # vertices = np.array([[float(x1), float(y1)], [float(x2), float(y2)], 
        #                      [float(x3), float(y3)]], np.float32)
        vertices = np.array([[float(x[0]), float(x[1])], [float(x[2]), float(x[3])], 
                             [float(x[4]), float(x[5])]], np.float32)
        
        sub_vertices = np.array([[float(x[6]), float(x[7])], [float(x[8]), float(x[9])], 
                             [float(x[10]), float(x[11])]], np.float32)
        
        triangles[i][:][:] = vertices*dim
        v_coords = np.int32((triangles[i][:][:]).reshape((-1,1,2)))

        sub_triangles[i][:][:] = sub_vertices*dim
        sub_v_coords = np.int32((sub_triangles[i][:][:]).reshape((-1,1,2)))


        cv2.polylines(img, [v_coords], True, (0, 0, 0))
        cv2.polylines(img, [sub_v_coords], True, (0, 0, 0))
        i=i+1
    cv2.imwrite('spidron_template.png', img)



# # Generation of seed points and triangulation

# ## The following generates random points within traingular region

for i  in range(8):
    with open('triangle{}.txt'.format(i), 'w') as f:
        perimeter_pts = perimeter_points(sub_triangles[i][:][:].reshape((-1,1,2)))
        points = [point_on_triangle(sub_triangles[i][:][:].reshape((-1,1,2))) for _ in range(3)]
        
        for point in points:
            f.write(f"{point[0]}\t{point[1]}\n")

        for point in perimeter_pts:
            f.write(f"{point[0][0]}\t{point[0][1]}\n")