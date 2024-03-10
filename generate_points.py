import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from mesh import draw_delaunay, draw_point, trisample, solid_triangle

##### Draw Semi-Spidron
# create white image
img = np.zeros((450,600,4), np.uint8)
# img.fill(255)

# first - Triangle 
first_pt1 = [50, 50]
first_pt2 = [50, 400]
first_pt3 = [400, 50]

# Points on the perimeter of first triangle
pts_first_1_2 = np.linspace(first_pt1, first_pt2, 5, axis=1)
pts_first_2_3 = np.linspace(first_pt2, first_pt3, 5, axis=1)
pts_first_3_1 = np.linspace(first_pt3, first_pt1, 5, axis=1)

# second - Triangle
second_pt1 = [400, 50]
second_pt2 = [225, 225]
second_pt3 = [575, 225]

# Points on the perimeter of second triangle
pts_sec_1_2 = np.linspace(second_pt1, second_pt2, 5, axis=1)
pts_sec_2_3 = np.linspace(second_pt2, second_pt3, 5, axis=1)
pts_sec_3_1 = np.linspace(second_pt3, second_pt1, 5, axis=1)


# third - triangle
third_pt1 = [575, 400]
third_pt2 = [575, 225]
third_pt3 = [400, 225]

# Points on the perimeter of the third triangle

pts_third_1_2 = np.linspace(third_pt1, third_pt2, 5, axis=1)
pts_third_2_3 = np.linspace(third_pt2, third_pt3, 5, axis=1)
pts_third_3_1 = np.linspace(third_pt3, third_pt1, 5, axis=1)


first_pts = np.array([first_pt1, first_pt2, first_pt3], np.int32)
first_pts = first_pts.reshape((-1,1,2))

cv2.polylines(img, [first_pts], True, (0, 0, 0, 255))

second_pts = np.array([second_pt1, second_pt2, second_pt3], np.int32)
second_pts = second_pts.reshape((-1,1,2))

cv2.polylines(img, [second_pts], True, (0, 0, 0, 255))



third_pts = np.array([third_pt1, third_pt2, third_pt3], np.int32)
third_pts = third_pts.reshape((-1,1,2))

cv2.polylines(img, [third_pts], True, (0, 0, 0, 255))

cv2.imwrite('triangle.png', img)


# random.seed(312345)

# # print(tuple(third_pts[0].ravel()))

# points = [trisample(first_pts) for _ in range(20)]


# with open('first_tri.txt', 'w') as f:
#     for point in points:
#         f.write(f"{np.int32(point)}\n")


# x, y = zip(*points)
# plt.scatter(x,y, s=4)
# plt.show()



# # Define colors for drawing.
# delaunay_color = (255, 255, 255, 255)
# points_color = (0, 0, 0, 255)
# fill_color = (0, 0, 0, 255)

# size = img.shape
# rect = (0, 0, size[1], size[0])
 
# # The following snippets performs Delaunay Triangulation
# # on the sub-traingles of semi-spidron

# for i in range(3):
#     # Create an instance of Subdiv2D
#     subdiv = cv2.Subdiv2D(rect)
    
#     # Create an array of points.
#     nodes = []
#     triangle_list = []

#     # Read in the points from a text file
#     with open("lasrtag{}.txt".format(i)) as file :
#         for line in file :
#             x, y = line.split()
#             nodes.append((int(x), int(y)))

#     # Insert points into subdiv
#     for p in nodes :
#         subdiv.insert(p)
    
    
#     ## Store the coordinates of the mesh triangles
#     # with open('triangle_list{}.txt'.format(i), 'w') as f:
#     #     for line in triangle_list:
#     #         f.write(f"{np.int32(line)}\n")

#     # Draw delaunay triangles
#     # draw_delaunay( img, subdiv, delaunay_color )
#     triangle_list = subdiv.getTriangleList()
#     idx = np.random.randint(0, len(triangle_list), 5)
#     solid_triangle(triangle_list[idx], img, fill_color)
    
#     # Draw points
#     # for p in nodes :
#     #     draw_point(img, p, points_color)
 
 
# # Show results
        
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
# # print(np.array(triangle_list).shape)       

# cv2.imwrite('tri_delaunay.png', img)




