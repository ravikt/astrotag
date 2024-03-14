# This script generates a complete sqaure spidron template

import numpy as np 
import cv2
import random
import math
from mesh import solid_triangle, generate_seed_points
from read_sig import calculate_centroid



# triangles, sub_triangles = generate_seed_points('astrotag_matrix.txt', img)



def create_marker(num_markers):

    for i in range(num_markers):
        img = np.zeros((dim,dim,3), np.uint8)
        img.fill(255)
        size = img.shape
        rect = (0, 0, size[1], size[0])
        # The following snippets performs Delaunay Triangulation
        # on the sub-traingles of semi-spidron
        nodes_list = ['triangle0.txt', 'triangle1.txt', 'triangle2.txt', 'triangle3.txt',
                    'triangle4.txt', 'triangle5.txt', 'triangle6.txt', 'triangle7.txt']

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
                    nodes.append((int(float(x)), int(float(y))))

            # Insert points into subdiv
            for p in nodes:
                # print(p)
                subdiv.insert(p)
                
            # Draw delaunay triangles
            # draw_delaunay( img, subdiv, delaunay_color )
            triangle_list = subdiv.getTriangleList()

            # Store the vertices of the mesh triangles
            with open('triangle_list{}.txt'.format(i), 'a') as f:
                for line in triangle_list:
                    l = np.int32(line)
                    print(l[0], l[1], l[5])
                    f.write(f"{np.int32(line)}\n")

            random.seed(312345)
            idx = np.random.randint(0, len(triangle_list), 5)
            solid_triangle(triangle_list[idx], img, fill_color)
            
            ## Draw points
            # for p in nodes :
            #     draw_point(img, np.int32(p), points_color)  

        sub_img = cv2.resize(img, (300, 300), interpolation= cv2.INTER_LINEAR)
        img[200:500, 200:500] = sub_img

        ## Draw the two Squares

        out_sq = np.array([[0, 0], [700, 0], [700, 700], [0, 700]], np.int32)
        out_sq = out_sq.reshape((-1,1,2))
        cv2.polylines(img, [out_sq], True, border_color, 16)


        in_sq = np.array([[175, 175], [525, 175], [525, 525], [175, 525]], np.int32)
        in_sq = in_sq.reshape((-1,1,2))
        cv2.polylines(img, [in_sq], True, border_color, 16)

        cv2.imwrite('marker_{}.png'.format(i), img)


if __name__ == "__main__":
    random.seed(312345)
    ##### Draw Semi-Spidron
    # create white image
    border_color = (0, 0, 0)
    fill_color = (0, 0, 0)
    delaunay_color = (0, 0, 0)
    points_color = (0, 0, 255)

    dim = 700

    create_marker(5)