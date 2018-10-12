import cv2
import numpy as np
import matplotlib.pyplot as plt


# def rle_encode(img): # run_length encde..
#     '''
#     img: numpy array, 1 - mask, 0 - background
#     Returns run length as string formated
#     '''
#     pixel = img.T.flatten()
#     pixels = np.concatenate([[0], pixels, [0]])
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
#     runs[1::2] -= runs[::2]
#     return ' '.join(str(x) for x in runs)
#
#
#
# def rle_decode(mask_rle, shape=(768, 768)):
#     '''
#     mask_rle: run-length as string formated (start length)
#     shape: (height,width) of array to return
#     Returns numpy array, 1 - mask, 0 - background
#     '''
#     s = mask_rle.split()
#     starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
#     starts -= 1
#     ends = starts + lengths
#     img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
#     for lo, hi in zip(starts, ends):
#         img[lo:hi] = 1
#     return img.reshape(shape).T  # Needed to align to RLE direction
#
#
# def masks_as_image(in_mask_list, all_masks=None):
#     # Take the individual ship masks and create a single mask array for all ships
#     if all_masks is None:
#         all_masks = np.zeros((768, 768), dtype = np.int16)
#     #if isinstance(in_mask_list, list):
#     for mask in in_mask_list:
#         if isinstance(mask, str):
#             all_masks += rle_decode(mask)
#     return np.expand_dims(all_masks, -1)







## create a mask of irregular shape... eg. rotated bounding box2
dir = "/Users/geewiz/Desktop/180928_periph/360periph/"              # 2nd set of experiments
image = cv2.imread(dir+'045.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

box = [[ 957, 1917], [1373 ,1810], [1804 ,1354], [1208, 825]]
box = np.array([box], dtype=np.int32)
mask = cv2.fillPoly(gray, box, 255)

index =(mask==255)
masked = image[index]
print(masked.shape)
#print(idx.shape)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# masked = gray[idx]
#masked = cv2.bitwise_and(image, mask)

plt.subplot(121), plt.imshow(mask)
plt.subplot(122), plt.imshow(masked)
plt.show()
