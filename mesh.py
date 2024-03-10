import cv2
import numpy as np
import random
import math 

def solid_triangle(tri_list, img, fill_color):
    for t in tri_list :
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        triangle_cnt = np.array([pt1, pt2, pt3])
        cv2.fillPoly(img, pts=[triangle_cnt], color=fill_color)
    # cv2.drawContours(img, [triangle_cnt], 0, (0,255,0), thickness=-1)

def uniform_points(v, n):
    '''
    Usage example
    v = np.array([(50,50), (50, 300), (300, 50)])
    v = np.array([(50,50), (50, 300), (300, 50)])
    points = uniform_points(v, 10)

    x, y = zip(*points)
    plt.scatter(x, y, s=1)
    plt.show()
    '''
    x = np.sort(np.random.rand(2, n), axis=0)
    return np.column_stack([x[0], x[1]-x[0], 1.0-x[1]])@v



def trisample(pts):
    '''
    Usage

    random.seed(312345)
    a = (50,50)
    b = (50,300)
    c = (300,50)
    points = [trisample(a,b,c) for _ in range(20)]

    print(points)

    x, y = zip(*points)
    plt.scatter(x,y, s=4)
    plt.show()
    '''
    
    a = tuple(pts[0].ravel())
    b = tuple(pts[1].ravel())
    c = tuple(pts[2].ravel())

    r1 = random.random()
    r2 = random.random()

    s1 = math.sqrt(r1)

    x = a[0]*(1.0-s1) + b[0]*(1.0-r2)*s1 + c[0]*r2*s1
    y = a[1]*(1.0-s1) + b[1]*(1.0-r2)*s1 + c[1]*r2*s1

    return(x,y)

# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True
 
# Draw a point
def draw_point(img, p, color ) :
    cv2.circle( img, p, 2, color, -1, cv2.LINE_AA, 0 )
 
# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :
 
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])
 
    for t in triangleList :
 
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        # print(pt1, pt2)
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
 
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
 
# Draw voronoi diagram
def draw_voronoi(img, subdiv) :
 
    (facets, centers) = subdiv.getVoronoiFacetList([])
    centers = np.int32(centers)
    for i in range(0,len(facets)) :
        ifacet_arr = []
        for f in facets[i] :
            ifacet_arr.append(f)
 
        ifacet = np.array(ifacet_arr, np.int32)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
 
        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0)
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), -1, cv2.LINE_AA, 0)
 
if __name__ == '__main__':
 
    # Define window names
    win_delaunay = "Delaunay Triangulation"
    win_voronoi = "Voronoi Diagram"
 
    # Turn on animation while drawing triangles
    animate = True
 
    # Define colors for drawing.
    delaunay_color = (255, 0, 0)
    points_color = (0, 0, 255)
 
    # Read in the image.
    img = cv2.imread("lasrtag.png")
 
    # Keep a copy around
    img_orig = img.copy()
 
    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])
 
    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect)
 
    # Create an array of points.
    points = []
 
    # Read in the points from a text file
    with open("points_lasrtag.txt") as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))
 
    # Insert points into subdiv
    for p in points :
        subdiv.insert(p)
 
        # Show animation
        if animate :
            img_copy = img_orig.copy()
            # Draw delaunay triangles
            draw_delaunay( img_copy, subdiv, (255, 255, 255) )
            cv2.imshow(win_delaunay, img_copy)
            cv2.waitKey(100)
 
    # Draw delaunay triangles
    draw_delaunay( img, subdiv, delaunay_color )
 
    # Draw points
    for p in points :
        draw_point(img, p, (0,0,255))
 
    # Allocate space for Voronoi Diagram
    img_voronoi = np.zeros(img.shape, dtype = img.dtype)
 
    # Draw Voronoi diagram
    draw_voronoi(img_voronoi,subdiv)
 
    # Show results
    cv2.imwrite('win_delaunay.png', img)
    cv2.imwrite('win_voronoi.png', img_voronoi)
