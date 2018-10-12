#!/bin/python3
# reference: https://sites.google.com/view/cvia/focus-measure

import cv2
import numpy as np
import glob
import pylab
import sys
from matplotlib import pyplot as plt
from scipy import stats
import csv

def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def yuv(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)
    return (img_yuv, y)

def crop(image):
    return image[950:1200, 1200:1680]   # y:y+dely, x:x+delx... EXP 1-5
#    return image[950:1200, 1200:1680]
#    cv2.imshow('cropped', im_cropped)
#    cv2.waitKey(5)

def crop_v2(image, x, y, delx, dely):
    return image[y:y+dely, x:x+delx]

# crops one image into two
def crop2(image):
    #return [image[950:1200, 1200:1500], image[1020:1150 ,1500:1650]]
    return [image[960:960+250, 1150:1150+350], image[960:960+250, 1500:1500+350]]


# prompts the user to draw a rectangle of roi... return r = [x, y, delx, dely]
def select_roi_crop(image):
    fromCenter = False
  #  showCrosshair = False
    height, width, channel = image.shape
    if height > 720:
         preview = cv2.resize(image, (0, 0), fx = 0.25, fy = 0.25)
    r = cv2.selectROI("select roi please...", preview, fromCenter)   # r = [x, y, delx, dely]
    x, y, delx, dely = (int(r[0]*4), int(r[1]*4), int(r[2]*4), int(r[3]*4))
    img_crop = crop_v2(image, x, y, delx, dely)
    rect = ((x+delx, y+dely), (delx, dely), 0)
    return img_crop


# DOES NOT WORK.. use scipy mouse click instead.. cv2.imshow() image is too large.
def capture_mouse_clicks(event, x, y, flags, param):
    global pts_list
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        param = (x, y)
        cv2.circle(img_roi, (x, y), 100, (0, 255, 0), -1)
        pts_list.append((x, y))



# given a folder, turn image to video
def make_gif(dir):
    w,h = 772, 519
#    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(dir+'output.avi',fourcc, 2.0, (w,h))
    # read in image
    for filename in sorted(glob.glob(dir+'*.png')):
        print(filename)
        # down size
        image = cv2.imread(filename)
        img_down = cv2.resize(image, (0, 0), fx = 0.25, fy = 0.25)
        # write frames to video
        out.write(img_down) # write flips the image
    print('video write completes')
#    cap.release()
    out.release()
    #cv2.



# UNTESTED: input an image and generate corners / contour of ROI
# results: not the best approach.. to find the roi
# INCOMPLETE compare if two contours are close enough, if so, merge



def find_roi(image):
    # find shape above certain brightness
    # lower = np.array([0, 0, 0])
    # upper = np.array([127, 127, 127])
    mask = cv2.inRange(image, 0, 128)  #image, lower, upper
    # detect shapes
    cnts, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#    print(len(contours), contours[0])
    # determine if contour is close, if so, merge
    lengthc = len(contours)
    status = np.zeros(lengthc)
    unified = []
    maximum = int(status.max())+1

    for i,cnt1 in enumerate(contours):
        x = i
        if i != lengthc-1:
            for j,cnt2 in enumerate(contours[i+1:]):
                x = x+1
                dist = cnts_are_close(cnt1,cnt2)
                if dist == True:
                    val = min(status[i],status[x])
                    status[x] = status[i] = val
                else:
                    if status[x]==status[i]:
                        status[x] = i+1
    #unify the contours
    for i in range(maximum):
        pos = np.where(status==i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(hull)
    # draw findContours
#    cv2.drawContous(image, [cnt], -1, (0, 255, 0), 3)
#    cv2.drawContours(image, contours, -1, (0,255,0), 2) # cv2.FILLED)
    print ("found %d contours, unified to %d" % (lengthc, len(unified)))
    cv2.drawContours(image,unified,-1,(0,255,0),2)

def gen_roi_list(image, angle):
 #   angles = range(0, 360, 15)
    rois = []

    theta = 0
    for theta in range (0, 360, angle):
        if theta > 360:
            break
        elif theta == 0:
            image = cv2.imread(image)
            roi = select_roi(image)
            c = r2c(roi)
            box = rot_roi(c, theta)
            rois.append(box)


# END OF INCOMPLETE FUNCTIONS



def laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F, 5)

def edge(img):
    sy = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    sx = cv2.Sobel(img, ddepth=cv2.CV_64F,dx=1, dy=0, ksize=5)
    return (np.hypot(sx, sy))

def edge_sq(img):
    sy = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    sx = cv2.Sobel(img, ddepth=cv2.CV_64F,dx=1, dy=0, ksize=5)
    return (np.power(np.hypot(sx, sy), 2))

def sobel_h(img):
    sobel_h = cv2.Sobel(img, ddepth=cv2.CV_64F,dx=0, dy=1, ksize=5)
#    sobel_v = cv2.Sobel(img, ddepth=cv2.CV_64F,dx=1, dy=0, ksize=5)
    return sobel_h #, sobel_v)

def sobel_v(img):
#    sobel_h = cv2.Sobel(img, ddepth=cv2.CV_64F,dx=0, dy=1, ksize=5)
    sobel_v = cv2.Sobel(img, ddepth=cv2.CV_64F,dx=1, dy=0, ksize=5)
    return sobel_v




def plot(y):
    t = np.arange(0, len(y))
    plt.plot(t, y)
    plt.ylabel('sharpness')
    plt.xlabel('frame #')
    plt.show()

def calc_diff(filename):
    image = cv2.imread(dir+filename)
    img_gray = gray(image)
    img_crop = crop(img_gray)
    line = np.float32(img_crop[100])
    grad = np.gradient(line, 2)        # gradient, 2nd order
    diff = np.diff(line)              # difference, 1st order **
    return [img_crop, line, grad, diff]

def test_filter(filename):
    image = cv2.imread(dir+filename)       # for EXP 1 & 2
#    image = cv2.imread(filename)            # for EXP 3
    img_gray = gray(image)
    img_crop = crop(img_gray)
    lap = laplacian(img_crop)
    edg = edge(img_crop)
    return (img_crop, lap, edg)

#    sqr = edge_sq(img_crop)
#    return [img_crop, lap, edg, sqr]

def test_depth(filename):
#    image = cv2.imread(dir+filename)            # for EXP 3
    image = cv2.imread(filename)                # EXP 5
#    cv2.imshow('test', image)
#    cv2.waitKey(5)
    img_gray = gray(image)
    img_1m, img_2m = crop2(img_gray)
    lap1 = laplacian(img_1m)
    lap2 = laplacian(img_2m)
    edg1 = edge(img_1m)
    edg2 = edge(img_2m)
    return (img_1m, img_2m, lap1, lap2, edg1, edg2)


def extract_frame(videoFile, fps):
    cap = cv2.VideoCapture(dir + videoFile)
    framerate = cap.get(cv2.CAP_PROP_FPS)  # grab a frame rate   18.24???!!!!
    print('current frame rate is %s' % framerate)
    x = 100  # file name start from 10.. to avoid zero fill

    while(cap.isOpened()):
        frameId = cap.get(1)    # current frame number
        ret, frame = cap.read()
        if ret != True :
            break
        if (frameId % (framerate//n) == 0):
            filename = dir + 'png/thumb%s.png' % str(int(x))
            x += 1
            cv2.imwrite(filename, frame)
    cap.release()
    print('frame extraction complete. extract %s frames' % x)


def solve_linear (img_xy, plt_xy):
    x = plt_xy[0][0], plt_xy[1][0]
    y = img_xy[0][0], img_xy[1][0]
    A = np.vstack([x, np.ones(2)]).T
    m, c = np.linalg.lstsq(A, y, -1)[0]

    x = plt_xy[0][1], plt_xy[1][1]
    y = img_xy[0][1], img_xy[1][1]
    B = np.vstack([x, np.ones(2)]).T
    n, d = np.linalg.lstsq(B, y, -1)[0]
    coeff = m, c, n, d
    return coeff

def solve_new_img_xy (unknown_plt_xy, coeff):
    m, c, n, d = coeff
    x = np.int(m*unknown_plt_xy[0]+c)
    y = np.int(n*unknown_plt_xy[1]+d)
    return x, y

# use the first two points to calibrate.. then the remaining points to create a bounding box
def solve_pts(pts):
    #extract the first two calibration points, plt_xy
    plt_xy = pts[0:2]
    img_xy = [[0, 0], [image.shape[1], image.shape[0]]]
    #print(plt_xy, img_xy)
    coeff = solve_linear(img_xy, plt_xy)

    for pt in pts[2:]:
        new_pt = solve_new_img_xy(pt, coeff)
        new_pts.append(new_pt)
    #print(new_pts)
    return new_pts

# given a point list, reduce the list using convexHull,
# make a bounding box, rotate it if neccessary
def create_bounding_box(image, pts):
    pts = np.array(pts)
    print(pts)
    pts_hull = cv2.convexHull(pts)      # fix convex
    rect = cv2.minAreaRect(pts_hull) #pts_hull)

    box = cv2.boxPoints(rect)   # box = [[], [], [], []]
    box = np.int0(box)
    cv2.polylines(image, [box], True, (0,255,0), thickness=20)
#    cv2.imshow('', image)
#    cv2.waitKey()
    print(angle)
    return (image, box)


# rotate the iamge, and create a straight bounding box
def create_bounding_box_crop(image, pts):
    pts = np.array(pts)
    pts_hull = cv2.convexHull(pts)      # fix convex
    rect = cv2.minAreaRect(pts_hull) #pts_hull)

    angle = rect[2]
    h, w = image.shape[0:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle+90, 1)
    img_rot = cv2.warpAffine(image, M, (w, h))

    img_crop = select_roi_crop(img_rot)
    return(img_crop)



def onclick(event):
#    print("(x, y) = %d, %d" % (event.x, event.y))
    print('click on ORIGIN and MAX to calibrate.. and then POINTS for roi.. press ENTER when done.')
    pts_list.append([event.x, event.y])
    print(pts_list)

def onkey(event):
    if event.key == 'enter':
        print('complete point selection..')
        plt.close()
#        return

class select_roi2:
    def __init__(self, pts_list):
        self.pts = pts_list
        self.press = None
        self.cid = self.fig.canvus.mpl_connect('button_press_event', self)

    # def __call__(self, event):
    #     print('click on ORIGIN and MAX to calibrate.. and then POINTS for roi')
    #     pts_list.append([event.x, event.y])
    #     print(pts_list)

    def connect(self):
        'connct to all the events'
        self.cidpress = self.fig.canvus.mpl_connect('button_press_event', self.onclick)
        self.cidclose = self.fig.canvus.mpl_connect('close_event', self.handle_close)

    def onclick(self, event):
    #    print("(x, y) = %d, %d" % (event.x, event.y))
        print('click on ORIGIN and MAX to calibrate.. and then POINTS for roi')
        pts_list.append([event.x, event.y])
        print(pts_list)

    def handle_close(self, event):
        plt.close()
        print("close plot...")

    def disconnect(self):
        'disconnect all stored connections'
        self.fig.canvas.mpl_disconnect(self.cidpress)
        self.fig.canvas.mpl_disconnect(self.cidclose)


if __name__ == "__main__":
#    dir = "/Users/geewiz/Desktop/180920_focus_test_seq/png_symm/"      # 1st set of experiments
#    dir = "/Users/geewiz/Desktop/180925_p3p4_1m/png_symm/"              # 2nd set of experiments
#    dir = "/Users/geewiz/Desktop/180928_periph/360periph/"              # 2nd set of experiments
#    dir = "/Users/geewiz/Desktop/180928_periph/195to0pass2/" #0to195pass1/"              # 2nd set of experiments
    dir = "/Users/geewiz/Desktop/180928_periph/360periph_sharpangle/"              # 2nd set of experiments

    images = []
    count = 0
    sharpness_max = [0, 0]
    sharpness_list = []
    bkgnd_list = []
    index = [0, 0]
    #global pts_list
    pts_list = []
    new_pts = []


    # raed in files, and make a rects[]

    # glob images, read in image,

    # image processing, rotate, gray, crop


    '''
    EXP 9: lens profile from peripheral to pheripheral

    RESULTS: very interesting results.. the lens is the sharpest at the centre (value of 600),
    while at the peripheral the sharpness is down to 250

    two things to try:
    1. chromatic correction
    2. curve fitting to normalize, assuming a sharpness of "1" from periph to periph
    '''
#
#     i = 0
#     H, W = (2076, 3088) # original image width and height
#
#     dir_csv = '/Users/geewiz/python/lab_cam_focus/'
#     filename = 'data_pass2.csv'
#
#     angle = 0
#     angles = []
#     rects = []
#
#
#     plt.subplots(2, 7)
#
#
#     # read in ROI list
#     with open(dir_csv+filename,'r')as f:
#         reader = csv.reader(f, delimiter=',')
#         header = next(reader)
#         for row in reader:
#             angle = int(row[0])
#             rect = (int(row[1]), int(row[2]), int(row[3]), int(row[4]))
#             angles.append(angle)
#             rects.append(rect)
#
#     # read in images and process
#     for filename in sorted(glob.glob(dir+'*.png')):
#         #if angle % 90 == 0:
#     #    print(angle)
#
#         image = cv2.imread(filename)
#         angle = angles[i]
#         # image processing, rotate, gray, crop
# #        M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1)
# #        rot = cv2.warpAffine(image, M, (W, H))
#         gry = gray(image)
#         crp = crop_v2(gry, rects[i][0], rects[i][1], rects[i][2], rects[i][3])
#         # plot
#         plt.subplot(2, 7, i+1), plt.imshow(crp), plt.title('%d' % angle)
#
#         # calculate sharpness
#         lap = laplacian(crp) # value
#         sharpness_l = lap.flatten().var()
#         edg = edge(crp) # value
#         sharpness_e = edg.flatten().var()
#
#         sharpness_list.append(sharpness_l)      # choose the method to calculate sharpness
# #        bkgnd_list.append(bkgnd_l)      # choose the method to calculate sharpness
# #        angles.append(angle)
#         #else: break
#
#         # update values
#         i += 1
#
#
#     # write sharpness_list, and then do curve fitting
#
#     plt.figure()
#     plt.plot(angles, sharpness_list, 'bo'), plt.title('sharpness vs angle - lap.')
# #    plt.text(20, 2000000, "stdev(sharpness) = %.2e and 3*stdev(bkgnd) = %.2e" % (std_i, 3* std_b))
#     plt.show()
#
#
#
#



    '''EXP 8 - compare sharpness curve at centre vs at periph, as lens turns.. see general shape.
   RESULTS:

    analyzed 60cm, 75cm, 90cm.. in general the curves at centre and peripheral are similar..

    60 cm is the distance to centre screen.. 90cm is further away
    distance to side screen (periph) is gradually towards the centre of the projection

    at 60cm centre peaks at 250, while periph at 100; bottoms at 10 - 20
    at 75cm centre peaks at 350, periph at 150, bottoms at 20
    at 90cm centre peaks at 300, periph at 225, bottoms at 0

    '''
#     sharpness_list_centre = []
#     sharpness_list_periph = []
#
#     angle = 0
#     angles = []
#     i = 1
#     H, W = (2076, 3088) # original image width and height
#
#     for filename in sorted(glob.glob(dir+'*.png')):
#         print(filename)
#         image = cv2.imread(filename)
#         # image processing, rotate, gray, crop
#         gry = gray(image)
# #        crp = crop_v2(gry, 1200, 1730, 600, 300) # cropping the bottom screen
#         crp_centre = crop_v2(gry, 1350, 900, 400, 250) # cropping the centre
# #        crp_centre = crop_v2(gry, 1250, 950, 600, 300)
# #        crp_periph = crop_v2(gry, 1270, 1770, 600, 300) # cropping bottom
#         crp_periph = crop_v2(gry, 500, 700, 300, 550)   # cropping left
# #        crp_periph = crop_v2(gry, 1200, 0, 600, 300)   # cropping top
#
#         lap_centre = laplacian(crp_centre) # value
#         sharpness_lap_centre = lap_centre.flatten().var()
#         edg_centre = edge(crp_centre) # value
#         sharpness_edg_centre = edg_centre.flatten().var()
#
#         lap_periph = laplacian(crp_periph) # value
#         sharpness_lap_periph = lap_periph.flatten().var()
#         edg_periph = edge(crp_periph) # value
#         sharpness_edg_periph = edg_periph.flatten().var()
#
#         sharpness_list_centre.append(sharpness_lap_centre)      # choose the method to calculate sharpness
#         sharpness_list_periph.append(sharpness_lap_periph)      # choose the method to calculate sharpness
#
#     # analyze sharpness values
# #    print(stats.describe(sharpness_list))
#     # std_i = np.std(sharpness_list)
#     # std_b = np.std(bkgnd_list)
#
#     plt.subplot(131), plt.imshow(crp_centre), plt.title('centre')
#     plt.subplot(132), plt.imshow(crp_periph), plt.title('peripheral')
#     plt.subplot(133), plt.plot(sharpness_list_centre, 'bo', sharpness_list_periph, 'r*')
#     plt.title('sharpness vs frames - lap.')
# #    plt.text(20, 2000000, "stdev(sharpness) = %.2e and 3*stdev(bkgnd) = %.2e" % (std_i, 3* std_b))
#     plt.legend(('centre', 'pheripheral'), loc = 'upper right')
#     plt.show()
#


    '''
    EXP 7c - rewrite..(cont'd) - examine only the multliples of 90, without rotation
    CONCLUSION:
    '''
    angle = 0
    angles = []
    i = 1
    H, W = (2076, 3088) # original image width and height
    rects = []

    plt.subplots(5, 5)
    dir_csv = '/Users/geewiz/python/lab_cam_focus/'
    csv_filename = 'data_360periph.csv'
    # read in ROI list
    with open(dir_csv+csv_filename,'r')as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for row in reader:
            angle = int(row[0])
            rect = (int(row[1]), int(row[2]), int(row[3]), int(row[4]))
            angles.append(angle)
            rects.append(rect)


    for filename in sorted(glob.glob(dir+'*.png')):
        print(filename, angles[i-1])
        image = cv2.imread(filename)
        gry = gray(image)
#        crp = crop_v2(gry, 1200, 1730, 600, 300) # cropping the bottom screen
        crp_i = crop_v2(gry, rects[i-1][0], rects[i-1][1], rects[i-1][2], rects[i-1][3])
        crp_b = crop_v2(gry, 1200, 850, 600, 300) # cropping the side for background analysis

        # plot
        plt.subplot(1, 5, i), plt.imshow(crp_i), plt.title('%d' % angles[i-1])

        # calculate sharpness
        lap_i = laplacian(crp_i) # value
        sharpness_l = lap_i.flatten().var()
        edg_i = edge(crp_i) # value
        sharpness_e = edg_i.flatten().var()

        lap_b = laplacian(crp_b) # value
        bkgnd_l = lap_b.flatten().var()
        edg_b = edge(crp_b) # value
        bkgnd_e = edg_b.flatten().var()

        sharpness_list.append(sharpness_l)      # choose the method to calculate sharpness
        bkgnd_list.append(bkgnd_l)      # choose the method to calculate sharpness
        #else: break

        # update values
        i += 1

    # analyze sharpness values
#    print(stats.describe(sharpness_list))
    std_i = np.std(sharpness_list)
    std_b = np.std(bkgnd_list)

    #
    # for j in range(len(angles)):
    #     #dict[angles[j]]=sharpness[j]
    #     dict.update({angles[j]:sharpness_list[j]})

    plt.figure()
#    plt.subplot(155)
    plt.plot(angles, sharpness_list, 'bo', angles, bkgnd_list, 'r*'), plt.title('sharpness vs angle - lap.')
    plt.text(0, 300, "stdev(sharpness) = %g and 3*stdev(bkgnd) = %g" % (std_i, 3* std_b))
    plt.legend(('centre', 'pheripheral'), loc = 'centre right')
    plt.show()



#
    '''
    EXP 7b - rewrite..
    CONCLUSION:
    1) the two methods (lapalcian and edge) give rather different trends...
    2) at 90, 180, 270, 360, and 0.. the results peak. expected, because image is not rotatedself.
    3) what's the tolerance? is it in fact focus difference or just variance?

    1) TEST W/ BACKGROUND - after plotting the background (arbituaray chosen as a the side of the image), the stdev(bkgnd) is 100x lower than stdev(sharp)..
    suggesting that it's the focus, not the variation...

    2) TEST W/ MULTIPLES OF 90 DEG - images captured at 90 degress do not undergo rotation during analysis
    yields much high sharpness values than those do rotate... however the trend around 360 are the masks_as_image

    another suspition is the image is warped, so the sharpness calculation become
    very sensitive to warping / projection angle / lumination...
    the sharpness values, whether using laplacian or edge vary by 10-20% of baseline, eg. 225 - 245 or 2.6M - 2.7M


    '''

#     angle = 0
#     angles = []
#     i = 1
#     H, W = (2076, 3088) # original image width and height
#
#     plt.subplots(5, 5)
#
#     for filename in sorted(glob.glob(dir+'*.png')):
#         print(filename)
#         #if angle % 90 == 0:
#     #    print(angle)
#
#         image = cv2.imread(filename)
#         # image processing, rotate, gray, crop
#         M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1)
#         rot = cv2.warpAffine(image, M, (W, H))
#         gry = gray(rot)
# #        crp = crop_v2(gry, 1200, 1730, 600, 300) # cropping the bottom screen
#         crp_i = crop_v2(gry, 1200, 0, 600, 300) # cropping the top screen
#         crp_b = crop_v2(gry, 2200, 700, 600, 300) # cropping the side for background analysis
#
#
# # try not to rotate 0, 90, 180, 270, 360
#
#
#
#
#         # plot
#         plt.subplot(5, 5, i), plt.imshow(crp_i), plt.title('%d' % angle)
#
#         # calculate sharpness
#         lap_i = laplacian(crp_i) # value
#         sharpness_l = lap_i.flatten().var()
#         edg_i = edge(crp_i) # value
#         sharpness_e = edg_i.flatten().var()
#
#         lap_b = laplacian(crp_b) # value
#         bkgnd_l = lap_b.flatten().var()
#         edg_b = edge(crp_b) # value
#         bkgnd_e = edg_b.flatten().var()
#
#         sharpness_list.append(sharpness_l)      # choose the method to calculate sharpness
#         bkgnd_list.append(bkgnd_l)      # choose the method to calculate sharpness
#         angles.append(angle)
#         #else: break
#
#         # update values
#         i += 1
#         angle += 15
#
#     # analyze sharpness values
# #    print(stats.describe(sharpness_list))
#     std_i = np.std(sharpness_list)
#     std_b = np.std(bkgnd_list)
#
#     #
#     # for j in range(len(angles)):
#     #     #dict[angles[j]]=sharpness[j]
#     #     dict.update({angles[j]:sharpness_list[j]})
#
#     plt.figure()
#     plt.plot(angles, sharpness_list, 'bo', angles, bkgnd_list, 'r*'), plt.title('sharpness vs angle - lap.')
#     plt.text(20, 2000000, "stdev(sharpness) = %.2e and 3*stdev(bkgnd) = %.2e" % (std_i, 3* std_b))
#     plt.show()


#
# # EXP 7 - lens peripheral, batch subprocess
#     for filename in sorted(glob.glob(dir+'*.png')):
#         print(filename)
#         fig, ax = plt.subplots()
#         image = cv2.imread(filename)
#
#         plt.imshow(image)
#         cid = fig.canvas.mpl_connect('button_press_event', onclick) # cid = call id
#         cid = fig.canvas.mpl_connect('key_press_event', onkey)
#         plt.show()
#
#         new_pts = solve_pts(pts_list)
#         crop = create_bounding_box_crop(image, new_pts)
#
#         gray = gray(crop)
#         #    edg = edge(gray) # value
#         #    sharpness = edg.flatten().var()
#         lap = laplacian(gray) # value
#         sharpness = lap.flatten().var()
#
#         plt.subplot(131), plt.imshow(image)
#         plt.subplot(132), plt.imshow(crop), plt.title('sharpness=%d' % sharpness)
#         #    plt.subplot(133), plt.plot(edge)
#         plt.show()
#
#         print('print Q')
#         sharpness_list.append(sharpness)
#         pts_list.clear()
#         new_pts.clear()
#
#
#     plt.subplot(133), plt.plot(sharpness_list)
#     plt.show()
#


#
# # EXP 7 - lens peripheral, single frame
#     fig, ax = plt.subplots()
#     image = cv2.imread(dir+'015.png')
#
#     plt.imshow(image)
#     cid = fig.canvas.mpl_connect('button_press_event', onclick) # cid = call id
#     cid = fig.canvas.mpl_connect('key_press_event', onkey)
#     plt.show()
#
#     new_pts = solve_pts(pts_list)
#     crop = create_bounding_box_crop(image, new_pts)
#
#     gray = gray(crop)
# #    edg = edge(gray) # value
# #    sharpness = edg.flatten().var()
#     lap = laplacian(gray) # value
#     sharpness = lap.flatten().var()
#
#     plt.subplot(121), plt.imshow(image)
#     plt.subplot(122), plt.imshow(crop), plt.title('sharpness=%d' % sharpness)
# #    plt.subplot(133), plt.plot(edge)
#     plt.show()




# # UNIT TEST OF capture_mouse_clicks
#     image = cv2.imread(dir+'045.png')
# #    preview = cv2.resize(image, (0, 0), fx = 0.25, fy = 0.25)
#

# #
# # UNIT TEST of create_bounding_box
#     image = cv2.imread(dir+'000.png')
#     pts = [[100, 100], [1000, 400], [30, 100], [50, 100], [50, 400], [80, 600]]
#     #pts = [[1290, 1835],[1500, 1865],[1690, 1845], [1745, 2010],[1500, 2050],[1255, 2010]]
#     img_roi, box = create_bounding_box(image, pts)
#     plt.imshow(img_roi)
#     plt.show()
#     #preview = cv2.resize(image, (0, 0), fx = 0.25, fy = 0.25)


# # UNIT TEST of rot_roi
#     box = []
#     image = cv2.imread(dir+'045.png')
# #    gray = gray(image)
#     r = select_roi(image)
#     box = rot_roi(r)
#     print(box)
#     img_cont = cv2.drawContours(image, [box], -1, (0, 255,0), 2)
#     cv2.imshow('', img_cont)
#     cv2.waitKey(0)

#    find_roi(resized)
    #print(stats.describe(gray.flatten()))

# # UNIT TEST of yuv
#     image = cv2.imread(dir+'000.png')
#     yuv, y = yuv(image)
#     gray = gray(image)
#     print(yuv.shape, y.shape, gray.shape)
#     print(sum(y-gray))
#     plt.subplot(121), plt.imshow(gray), plt.title('gray sacale')
#     plt.subplot(122), plt.imshow(y), plt.title('y')
#     plt.show()

## UNIT TEST of make_gif
#    make_gif(dir)

# # UNIT TEST of select_roi function
#     image = cv2.imread(dir+"p3p4_focus.png")
#     select_roi(image)




#print(rois)

    #
    # theta = 0
    # for theta in range (0, 360, angle)
    #     if theta > 360:
    #         break
    #     else theta == 0:
    #         image = cv2.imread(image)
    #         roi = select_roi(image)
    #         c = r2c(roi)
    #         box = rot_roi(c, theta)
    #         rois.append(box)




# # EXP6 comparison between gray vs Y images
# # RESULTS: gray and
#     image = cv2.imread(dir+'000.png')
#     img_crop = crop(image)
#     yuv, img_y = yuv(img_crop)
#     img_g = gray(img_crop)
#     sh_g = edge(img_g).var()
#     sh_y = edge(img_y).var()
#     print(sh_g, sh_y)
#
#     plt.subplot(211), plt.imshow(img_g), plt.title('gray var=%s' % sh_g)
#     plt.subplot(212), plt.imshow(img_y), plt.title('y ch. var=%s' % sh_y)
#     plt.show()


# # EXP5 compare pattern at 1m vs pattern at 2m.. they need to be on the same type of monitor, with the same setting...
#     img1m, img2m, lap1, lap2, edg1, edg2 = test_depth('p3p4_focus.png')
#
#     print(stats.describe(lap1.flatten()))
#     print(stats.describe(lap2.flatten()))
#
#     plt.subplot(231), plt.imshow(img1m), plt.title("at 1m")
#     plt.subplot(232), plt.imshow(lap1),
#
#     plt.subplot(234), plt.imshow(img2m), plt.title("at 2m")
#     plt.subplot(235), plt.imshow(lap2),
#     plt.show()



# # EXP5 compare pattern at 1m vs pattern at 2m.. they need to be on the same type of monitor, with the same setting...
#     img1m, img2m, lap1, lap2, edg1, edg2 = test_depth('p3p4_focus.png')
#
#     print(stats.describe(lap1.flatten()))
#     print(stats.describe(lap2.flatten()))
#
#     plt.subplot(231), plt.imshow(img1m), plt.title("at 1m")
#     plt.subplot(232), plt.imshow(lap1),
#
#     plt.subplot(234), plt.imshow(img2m), plt.title("at 2m")
#     plt.subplot(235), plt.imshow(lap2),
#     plt.show()


# EXP5b ... depth test... batch process the files to see if peak sharp appear at the same place for pattern at 1m and 2m..
# RESULTS: experimented with p3 and p4 resembles each other, p1 rather different...
# ... p1 and p3 are only off by a couple of frames, but p4 off by 4 frames.
# DISCUSSION: this experiement needs to be repeated using monitor of the same type?? try more patterns??
#
# EXP5c ... pattern comparison... recaptured video, projected different patterns side by side, on same type of monitor
# RESULTS: the two sharpness curve has a good resemblance, both are equally good to be used for focus measurement
# #
#     for filename in sorted(glob.glob(dir+'*.png')):
#         print(filename)
#         img1m, img2m, lap1, lap2, edg1, edg2 = test_depth(filename)
#         sharpness1m = lap1.flatten().var()
#         sharpness2m = lap2.flatten().var()
# #        sharpness1m = edg1.flatten().var()
# #        sharpness2m = edg2.flatten().var()
#         print(sharpness1m, sharpness2m)
#         sharpness_list[0].append(sharpness1m)
#         sharpness_list[1].append(sharpness2m)
#         count = count +1
#         if sharpness1m > sharpness_max[0]:
#             sharpness_max[0] = sharpness1m
#             index[0] = count
#         if sharpness2m > sharpness_max[1]:
#             sharpness_max[1] = sharpness2m
#             index[1] = count
#         if count == 120:
#             break
#     print(index[0], index[1])
#
#     fig, ax1 = plt.subplots()
#     ax2 = ax1.twinx()
#
#     ax1.plot(sharpness_list[0], 'go', label='sharpness at 1m')
#     ax2.plot(sharpness_list[1], 'rs', label='sharpness at 2m')
#     ax1.set_xlabel('frames')
# #    ax1.set_ylabel('sharpness at 1m [AU]', color = 'g')
# #    ax2.set_ylabel('sharpness at 2m [AU]', color = 'r')
# #    plt.title('comparison of sharpness at different distance (setting: -pattern 3 -ss 2 -fps 30)')
# #    plt.text(1, 300, 'lens is focused at frame %s at 1m, and %s at 2m' % (str(index[0]), str(index[1]) ))
#     ax1.set_ylabel('sharpness (sine wave) [AU]', color = 'g')
#     ax2.set_ylabel('sharpness (slanted sqr) [AU]', color = 'r')
#     plt.title('comparison of patterns (setting: -ss 2 -fps 2)')
#     plt.text(1, 300, 'lens is focused at frame %s for sine wave, and %s for slanted sqr' % (str(index[0]), str(index[1]) ))
#
#     ax1.ticklabel_format(style='sci', axis='Y', scilimits=(0,0))
#     ax2.ticklabel_format(style='sci', axis='Y', scilimits=(0,0))
#     plt.show()
#



# #   EXP4: comparing edge vs laplacian approach
# #   read in images in batches
# # RESULTS: the two methods yield sharpness curve tha resemble each other,
# # frames are off by 1 for pattern 3; no difference for patter 1.. believe either method is valid
# #
# # one interesting observation is the sharpness varies with pattern..
#
#     for filename in sorted(glob.glob(dir+'*.png')):
#         print(filename)
#         img_crop, lap, edg = test_filter(filename)
#         sharpness_l = lap.flatten().var()
#         sharpness_e = edg.flatten().var()
#         print(sharpness_l, sharpness_e)
#         sharpness_list[0].append(sharpness_l)
#         sharpness_list[1].append(sharpness_e)
#         count = count +1
#         if sharpness_l > sharpness_max[0]:
#             sharpness_max[0] = sharpness_l
#             index[0] = count
#         if sharpness_e > sharpness_max[1]:
#             sharpness_max[1] = sharpness_e
#             index[1] = count
#     #    if count == 120:
#     #        break
#     #print(index[0], index[1])
#
#     fig, ax1 = plt.subplots()
#     ax2 = ax1.twinx()
#
#     ax1.plot(sharpness_list[0], 'go', label='laplacian')
#     ax2.plot(sharpness_list[1], 'rs', label='edge det')
#     ax1.set_xlabel('frames')
#     ax1.set_ylabel('sharpness by Laplacian [AU]', color = 'g')
#     ax2.set_ylabel('sharpness by Edge detection [AU]', color = 'r')
#     plt.title('comparison of sharpness calcluation method (setting: -pattern 4 -ss 4 -t 5 -fps 30)')
#     plt.text(1, 310000, 'lens is focused at frame %s for lap, and %s for edge detection' % (str(index[0]), str(index[1]) ))
#     plt.ticklabel_format(style='sci', axis='Y', scilimits=(0,0))
#     plt.show()




# #   EXP3: process images in batches and plot sharpness
# #   read in images in batches
#     for filename in sorted(glob.glob(dir+'*.png')):
#         print(filename)
#         img_crop, lap, edg = test_filter(filename)
#         #sharpness = edg.flatten().var()
#         sharpness = lap.flatten().var()
#         print(sharpness)
#         sharpness_list.append(sharpness)
#         count = count +1
#         if sharpness > sharpness_max:
#             sharpness_max = sharpness
#             index = count
#
#     plt.plot(sharpness_list), plt.title('sharpness plot for pattern 3 (setting: -ss 3, -fps 12, -f lap)')
#     plt.ylabel('sharpness [AU]'), plt.xlabel('frames')
#     plt.text(2, 40, 'lens is focused at frame %s with a sharpness value of %s' % (str(index), str(np.round(sharpness_max, 2))))
#     plt.ticklabel_format(style='sci', axis='Y', scilimits=(0,0))
#     plt.show()
# #




# # EXP2: test filter, show line transfer function..
# # file folder /png_pattern
# # RESULTS: Laplacian shows a 10 fold difference in variance; edge detection, 6 fold
#     img_blur, lap_blur, edge_blur = test_filter('p4blur.png')
#     img_focus, lap_focus, edge_focus = test_filter('p4focus.png')
#
#     print(stats.describe(lap_blur.flatten()))
#     print(stats.describe(lap_focus.flatten()))
#     print(stats.describe(edge_blur.flatten()))
#     print(stats.describe(edge_focus.flatten()))
#     #print(stats.describe(sqr_blur.flatten()))
#     #print(stats.describe(sqr_focus.flatten()))
#
#     plt.subplot(231), plt.imshow(img_blur)
#     plt.subplot(232), plt.imshow(lap_blur), plt.title("laplacian")
#     plt.subplot(233), plt.imshow(edge_blur), plt.title("edge detection")
#
#     plt.subplot(234), plt.imshow(img_focus)
#     plt.subplot(235), plt.imshow(lap_focus), plt.title("laplacian")
#     plt.subplot(236), plt.imshow(edge_focus), plt.title("edge detection")
#
#     plt.show()
#
#


# # EXP1: calc difference, show line transfer function..
# # file folder /png_pattern
# # RESULTS: both methods showing 4 fold difference.. can use either one.
#     img_blur, line_blur, grad_blur, diff_blur = calc_diff('p3blur.png')
#     img_focus, line_focus, grad_focus, diff_focus = calc_diff('p3focus.png')
#
#     print(stats.describe(grad_blur), stats.describe(diff_blur))
#     print(stats.describe(grad_focus), stats.describe(diff_focus))
