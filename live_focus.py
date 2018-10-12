#from pyueye import ueye


import ueye

cam = ueye.cam()
image = cam.GrabImage()
imshow(image)
