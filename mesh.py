import cv2
import numpy as np
import random
import math 

def perimeter_points(tri):
    
    a = np.linspace(tri[0], tri[1], 3, axis=1)
    b = np.linspace(tri[1], tri[2], 3, axis=1)
    c = np.linspace(tri[2], tri[0], 3, axis=1)
    peri_pts = np.vstack((a,b,c)).reshape((-1,1,2))
    tuple(map(tuple, peri_pts))
    return peri_pts

def solid_triangle(tri_list, img, fill_color):
    for t in tri_list :
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        triangle_cnt = np.array([pt1, pt2, pt3])
        cv2.fillPoly(img, pts=[triangle_cnt], color=fill_color)
    # cv2.drawContours(img, [triangle_cnt], 0, (0,255,0), thickness=-1)

def point_on_triangle(pts):
    '''
    Source: https://stackoverflow.com/questions/47410054/
    Random point on the triangle with vertices pt1, pt2 and pt3.
    '''
    pt1 = tuple(pts[0].ravel())
    pt2 = tuple(pts[1].ravel())
    pt3 = tuple(pts[2].ravel())
    
    # x, y = random.random(), random.random()
    x, y = np.random.randn(), np.random.randn()
    q = abs(x - y)
    s, t, u = q, 0.5 * (x + y - q), 1 - 0.5 * (q + x + y)
    return (
        s * pt1[0] + t * pt2[0] + u * pt3[0],
        s * pt1[1] + t * pt2[1] + u * pt3[1],
    )



def trisample(pts):
    '''
    Source: https://stackoverflow.com/questions/47410054/
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
    # random.seed(312345)
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
    # sub_p = np.int32(((p/700)*300)+185)


# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :
 
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])
 

    for t in triangleList:
 
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        # print(pt1, pt2)
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
 
            cv2.line(img, pt1, pt2, delaunay_color, 2, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 2, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 2, cv2.LINE_AA, 0)


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
 

def generate_seed_points(filename, img):
    dim = img.shape[0]
    image = img.copy()
    triangles     = np.ones((12,3,2)) # used for template
    sub_triangles = np.ones((12,3,2)) # Used for encoding

    with open(filename) as file:
        i = 0
        for line in file:
            x = line.split()

            vertices = np.array([[float(x[0]), float(x[1])], [float(x[2]), float(x[3])], 
                                [float(x[4]), float(x[5])]], np.float32)
            
            sub_vertices = np.array([[float(x[6]), float(x[7])], [float(x[8]), float(x[9])], 
                             [float(x[10]), float(x[11])]], np.float32)
        
            triangles[i][:][:] = vertices*dim
            v_coords = np.int32((triangles[i][:][:]).reshape((-1,1,2)))

            sub_triangles[i][:][:] = sub_vertices*dim
            sub_v_coords = np.int32((sub_triangles[i][:][:]).reshape((-1,1,2)))


            cv2.polylines(image, [v_coords], True, (0, 0, 0))
            cv2.polylines(image, [sub_v_coords], True, (0, 0, 0))
            i=i+1
        cv2.imwrite('spidron_template.png', image)




    # # Generation of seed points and triangulation

    # ## Uncomment the below to generate seed points for all the 12 triangles

        for i  in range(8):
            with open('triangle{}.txt'.format(i), 'w') as f:
                perimeter_pts = perimeter_points(sub_triangles[i][:][:].reshape((-1,1,2)))
                points = [trisample(sub_triangles[i][:][:].reshape((-1,1,2))) for _ in range(3)]
                
                for point in points:
                    f.write(f"{point[0]}\t{point[1]}\n")

                for point in perimeter_pts:
                    f.write(f"{point[0][0]}\t{point[0][1]}\n")
    
    return triangles, sub_triangles