import cv2
import numpy as np
import random
import math 


def barycentric_points(pts):
    pt1 = tuple(pts[0].ravel())
    pt2 = tuple(pts[1].ravel())
    pt3 = tuple(pts[2].ravel())

    p1 = np.array([0.5, 0.25, 0.25])
    p2 = np.array([0.25, 0.5, 0.25])
    p3 = np.array([0.25, 0.25, 0.5])
    p4 = np.array([0.33, 0.33, 0.33])
    x = np.array([pt1[0], pt2[0], pt3[0]])
    y = np.array([pt1[1], pt2[1], pt3[1]])

    a = (np.dot(p1,x), np.dot(p1,y))
    b = (np.dot(p2,x), np.dot(p2,y))
    c = (np.dot(p3,x), np.dot(p3,y))
    d = (np.dot(p4,x), np.dot(p4,y))

    return np.vstack((a,b,c,d))