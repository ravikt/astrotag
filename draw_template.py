# This script generates a complete sqaure spidron template

import numpy as np 
import cv2
import random
import math
from mesh import draw_delaunay, draw_point, trisample, solid_triangle


##### Draw Semi-Spidron
# create white image
img = np.zeros((800,800,4), np.uint8)
# img.fill(255)




tri_11 = np.array([[50, 50], [400, 50], [50, 400]], np.int32)
tri_11 = tri_11.reshape((-1,1,2))

tri_12 = np.array([[400, 50], [750, 50], [750, 400]], np.int32)
tri_12 = tri_12.reshape((-1,1,2))

tri_13 = np.array([[750, 400], [750, 750], [400, 750]], np.int32)
tri_13 = tri_13.reshape((-1,1,2))

tri_14 = np.array([[50, 400], [400, 750], [50, 750]], np.int32)
tri_14 = tri_14.reshape((-1,1,2))

tri_21 = np.array([[50, 400], [225, 225], [225, 575]], np.int32)
tri_21 = tri_21.reshape((-1,1,2))

tri_22 = np.array([[225, 225], [400, 50], [575, 225]], np.int32)
tri_22 = tri_22.reshape((-1,1,2))

tri_23 = np.array([[575, 225], [750, 400], [575, 575]], np.int32)
tri_23 = tri_23.reshape((-1,1,2))

tri_24 = np.array([[575, 575], [400, 750], [225, 575]], np.int32)
tri_24 = tri_24.reshape((-1,1,2))

tri_31 = np.array([[225, 225], [400, 225], [225, 400]], np.int32)
tri_31 = tri_31.reshape((-1,1,2))

tri_32 = np.array([[400, 225], [575, 225], [575, 400]], np.int32)
tri_32 = tri_32.reshape((-1,1,2))

tri_33 = np.array([[575, 400], [575, 575], [400, 575]], np.int32)
tri_33 = tri_33.reshape((-1,1,2))

tri_34 = np.array([[400, 575], [225, 575], [225, 400]], np.int32)
tri_34 = tri_34.reshape((-1,1,2))


triangle_list = [tri_11, tri_12, tri_13, tri_14, tri_21, tri_22, 
                 tri_23, tri_24, tri_31, tri_32, tri_33, tri_34]

##Uncomment below for draing the template
# for triangle in triangle_list:
#     cv2.polylines(img, [triangle], True, (0, 0, 0, 255))
#     print(str(triangle))

# cv2.imwrite('spidron_template.png', img)

## Draw the two Squares

out_sq = np.array([[50, 50], [750, 50], [750, 750], [50, 750]], np.int32)
out_sq = out_sq.reshape((-1,1,2))

cv2.polylines(img, [out_sq], True, (0, 0, 0, 255), 16)


in_sq = np.array([[225, 225], [575, 225], [575, 575], [225, 575]], np.int32)
in_sq = in_sq.reshape((-1,1,2))
cv2.polylines(img, [in_sq], True, (0, 0, 0, 255), 8)


def perimeter_points(tri):
    a = np.linspace(tri[0], tri[1], 3, axis=1)
    b = np.linspace(tri[1], tri[2], 3, axis=1)
    c = np.linspace(tri[2], tri[0], 3, axis=1)
    peri_pts = np.vstack((a,b,c)).reshape((-1,1,2))
    tuple(map(tuple, peri_pts))
    return peri_pts

# Generation of seed points and traingulation

nodes_list = ['tri_11.txt', 'tri_12.txt', 'tri_13.txt', 'tri_14.txt', 'tri_21.txt', 'tri_22.txt', 
              'tri_23.txt', 'tri_24.txt', 'tri_31.txt', 'tri_32.txt', 'tri_33.txt', 'tri_34.txt']



## Uncomment the below to generate seed points for all the 12 traingles

# for seeds, triangle in zip(nodes_list, triangle_list):

#     with open(seeds, 'w') as f:
#         perimeter_pts = perimeter_points(triangle)

#         points = [trisample(triangle) for _ in range(3)]

#         for point in points:
#             f.write(f"{np.int32(point)}\n")

#         f.write(f"{np.int32(perimeter_pts)}\n")

# Define colors for drawing.

delaunay_color = (255, 0, 0, 255)
points_color = (0, 0, 255, 255)
fill_color = (0, 0, 0, 255)

# delaunay_color = (255, 255, 255, 255)
# points_color = (0, 0, 0, 255)
# fill_color = (0, 0, 0, 255)

size = img.shape
rect = (0, 0, size[1], size[0])

# The following snippets performs Delaunay Triangulation
# on the sub-traingles of semi-spidron
i=1
for seeds in nodes_list:
    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect)
    
    # Create an array of points.
    nodes = []
    triangle_list = []

    # Read in the points from a text file
    with open(seeds) as file :
        for line in file :
            x, y = line.split()
            nodes.append((int(x), int(y)))

    # Insert points into subdiv
    for p in nodes :
        subdiv.insert(p)
    
    triangle_list = subdiv.getTriangleList()
    
    # Store the coordinates of the mesh triangles
    with open('triangle_list{}.txt'.format(i), 'w') as f:
        for line in triangle_list:
            f.write(f"{np.int32(line)}\n")

    i = i+1
    # Draw delaunay triangles
    # draw_delaunay( img, subdiv, delaunay_color )
    triangle_list = subdiv.getTriangleList()
    idx = np.random.randint(0, len(triangle_list), 3)
    solid_triangle(triangle_list[idx], img, fill_color)
    
    # ## Draw points
    # for p in nodes :
    #     draw_point(img, p, points_color)
 
 
# Show results
        
# img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
# print(np.array(triangle_list).shape)       

cv2.imwrite('tri_delaunay.png', img)
