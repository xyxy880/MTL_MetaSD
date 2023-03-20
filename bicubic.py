import cv2
import imageio
img = cv2.imread('/hdd/tianchuan/sst1.png')
img2 = cv2.imread('/hdd/tianchuan/sst2.png')

height, width, c = img.shape

for i in range(height):
    for j in range(width):
        for k in range(c):
        # img[i,j] is the RGB pixel at position (i, j)
        # check if it's [0, 0, 0] and replace with [255, 255, 255] if so
            if img[i, j,k].sum() == 0:
                img[i, j] = [255, 255, 255]

imageio.imsave('/hdd/tianchuan/Meteorological_data/PhIRE/NoMTL1/test.png',img)
