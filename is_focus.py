#!/bin/python3
# reference: https://sites.google.com/view/cvia/focus-measure

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy import stats

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

def crop_v2(x, y, delx, dely):
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
    img_crop = crop_v2(x, y, delx, dely)
    rect = ((x+delx, y+dely), (delx, dely), 0)
    return img_crop


def capture_mouse_clicks(event, x, y, flags, param):
    global pts_list
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        param = (x, y)
        cv2.circle(img_roi, (x, y), 100, (0, 255, 0), -1)
        pts_list.append((x, y))


# given a point list, reduce the list using convexHull, make a bounding box, rotate it if neccessary
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


# given a folder, turn image to video
def make_gif(dir):
    w,h = 772, 519
#    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(dir+'output.avi',fourcc, 20.0, (w,h))
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

if __name__ == "__main__":
#    dir = "/Users/geewiz/Desktop/180920_focus_test_seq/png_symm/"      # 1st set of experiments
#    dir = "/Users/geewiz/Desktop/180925_p3p4_1m/png_symm/"              # 2nd set of experiments
    dir = "/Users/geewiz/Desktop/180928_periph/360periph/"              # 2nd set of experiments

    images = []
    count = 0
    sharpness_max = [0, 0]
    sharpness_list = [[], []]
    index = [0, 0]


    # EXP7
    image = cv2.imread(dir+'015.png')
#    img_roi = select_roi_crop(image)

    cv2.setMouseCallback('roi', capture_mouse_clicks)

    while True:
        cv2.imshow('roi', image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows


    # theta = 0
    # rois = []
    #

    # # read in image
    # for filename in sorted(glob.glob(dir+'*.png')):
    #     print(filename)
    #     image = cv2.imread(filename)
    # #    gray = gray(image)
    #     x, y, delx, dely = select_roi(image)
    #     #c = r2c(roi)
    #     box = rot_roi(x, y, delx, dely, theta)
    #     theta =+ 15
    #     img_roi = cv2.drawContours(image, [box], -1, (0, 255,0), 2)
    #     cv2.imshow('preview', img_roi)
    #     cv2.waitKey(10)
    #     rois.append(box)
    #     # print(box)




# # UNIT TEST OF capture_mouse_clicks
#     image = cv2.imread(dir+'045.png')
# #    preview = cv2.resize(image, (0, 0), fx = 0.25, fy = 0.25)
#



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

# UNIT TEST of make_gif
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
