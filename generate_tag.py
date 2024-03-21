# This script generates a complete sqaure spidron template
import random
import numpy as np
import cv2
from mesh import solid_triangle, generate_seed_points, draw_delaunay


img = np.zeros((700,700,3), np.uint8)
img.fill(255)

# triangles, sub_triangles = generate_seed_points('astrotag_matrix.txt', img)
def calculate_centroid(p):

    centroid = ((p[0]+p[2]+p[4])/3, (p[1]+p[3]+p[5])/3)
    return centroid


def create_marker(num_markers):
    DIM = 700
    for i in range(num_markers):
        img = np.zeros((DIM, DIM, 3), np.uint8)
        b_img=np.zeros((DIM, DIM, 3), np.uint8)
        img.fill(255)
        size = img.shape
        rect = (0, 0, size[1], size[0])
        # The following snippets performs Delaunay Triangulation
        # on the sub-traingles of semi-spidron
        nodes_list = ['triangle0.txt', 'triangle1.txt', 'triangle2.txt', 'triangle3.txt',
                    'triangle4.txt', 'triangle5.txt', 'triangle6.txt', 'triangle7.txt']
        # j=0
        for seeds in nodes_list:
            # Create an instance of Subdiv2D
            subdiv = cv2.Subdiv2D(rect)
            # Create an array of points.
            nodes = []
            triangle_list = []

            # Read in the points from a text file
            with open('astrotag/'+seeds) as file :
                for line in file :
                    x, y = line.split()
                    nodes.append((int(float(x)), int(float(y))))

            # Insert points into subdiv
            for p in nodes:
                # print(p)
                subdiv.insert(p)
                
            # Draw delaunay triangles
            draw_delaunay( img, subdiv, delaunay_color )
            triangle_list = subdiv.getTriangleList()

            # Store the vertices of the mesh triangles
            if (i == 0):
                with open('triangle_list.txt', 'a') as f:
                    for line in triangle_list:
                        c = calculate_centroid(np.float32(line))
                        f.write(f"{np.int32(line)}\t{np.int32(c)}\n")
            # j=j+1
            random.seed(312345)
            idx = np.random.randint(0, len(triangle_list), 5)
            solid_triangle(triangle_list[idx], img, fill_color)
            solid_triangle(triangle_list[idx], b_img, (255, 255, 255))

            
            ## Draw points
            # for p in nodes :
            #     draw_point(img, np.int32(p), points_color)  

        sub_img = cv2.resize(img, (300, 300), interpolation= cv2.INTER_LINEAR)
        img[200:500, 200:500] = sub_img

        b_sub_img = cv2.resize(b_img, (300, 300), interpolation= cv2.INTER_LINEAR)
        b_img[200:500, 200:500] = b_sub_img


        ## Draw the two Squares

        out_sq = np.array([[0, 0], [700, 0], [700, 700], [0, 700]], np.int32)
        out_sq = out_sq.reshape((-1,1,2))
        cv2.polylines(img, [out_sq], True, border_color, 10)
        cv2.polylines(b_img, [out_sq], True, (255,255,255), 10)



        in_sq = np.array([[175, 175], [525, 175], [525, 525], [175, 525]], np.int32)
        in_sq = in_sq.reshape((-1,1,2))
        cv2.polylines(img, [in_sq], True, border_color, 10)
        cv2.polylines(b_img, [in_sq], True, (255,255,255), 10)


        cv2.imwrite('thesis_marker_{}.png'.format(i), img)
        cv2.imwrite('thesis_b_marker_{}.png'.format(i), b_img)



if __name__ == "__main__":
    random.seed(312345)
    ##### Draw Semi-Spidron
    # create white image
    border_color = (0, 0, 0)
    fill_color = (0, 0, 0)
    delaunay_color = (18, 18, 222)
    points_color = (0, 0, 255)


    create_marker(1)