
import numpy as np
import cv2
from matplotlib  import pyplot as plt

# given two points on plt_xy, calculate the transfer function to two points on img_xy
# plt_xy = [[ox, oy], [cx, cy]]
def solve_linear (img_xy, plt_xy):
    x = plt_xy[0][0], plt_xy[1][0]
    y = img_xy[0][0], img_xy[1][0]
    A = np.vstack([x, np.ones(2)]).T
    m, c = np.linalg.lstsq(A, y, -1)[0]

    x = plt_xy[0][1], plt_xy[1][1]
    y = img_xy[0][1], img_xy[1][1]
    B = np.vstack([x, np.ones(2)]).T
    n, d = np.linalg.lstsq(B, y, -1)[0]

    coeff = (m, c, n, d)
    return coeff

def solve_new_img_xy (unknown_plt_xy, coeff):
    m, c, n, d = coeff
    x = np.int(m*unknown_plt_xy[0]+c)
    y = np.int(n*unknown_plt_xy[1]+d)
    return [x, y]


# given a point list, reduce the list using convexHull,
# make a bounding box, rotate it if neccessary
def create_bounding_box(image, pts):
    pts = np.array(pts)
    pts_hull = cv2.convexHull(pts)      # fix convex
    rect = cv2.minAreaRect(pts_hull)
    box = cv2.boxPoints(rect)   # box = [[], [], [], []]
    box = np.int0(box)
    cv2.polylines(image, [box], True, (0,255,0), thickness=50)
#    cv2.imshow('', image)
#    cv2.waitKey()
    return (image, box)


def solve_pts(pts):
    new_pts = []
    #extract the first two calibration points, plt_xy
    plt_xy = pts[0:2]
    img_xy = [[0, 0], [image.shape[1], image.shape[0]]]
    #print(plt_xy, img_xy)
    coeff = solve_linear(img_xy, plt_xy)

    for pt in pts[3:]:
        new_pt = solve_new_img_xy(pt, coeff)
        new_pts.append(new_pt)
    #print(new_pts)
    return new_pts

# # UNIT TEST of linear solver
# img_xy = [[0, 0], [3000, 2000]]
# plt_xy = [[83, 398], [576, 73]]
# print(solve_new_img_xy([90, 90]))

def box2boundry(box):
    rect = top, bottom, left, right
    return rect

# UNIT TEST of create bounding box2
dir = "/Users/geewiz/Desktop/180928_periph/360periph/"              # 2nd set of experiments
pts = [[0, 1000], [1000, 0], [100, 100], [1000, 400], [30, 100], [50, 100], [50, 400], [80, 600]]
#pts = [[1290, 1835],[1500, 1865],[1690, 1845], [1745, 2010],[1500, 2050],[1255, 2010]]
image = cv2.imread(dir+'045.png')
new_pts = solve_pts(pts)
image, box = create_bounding_box(image, new_pts)
