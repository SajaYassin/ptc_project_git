

import matplotlib.pyplot as plt
import cv2


def shade(imag, percent):
    """
    imag: the image which will be shaded
    percent: a value between 0 (image will remain unchanged
             and 1 (image will be blackened)
    """
    tinted_imag = imag * (1 - percent)
    return tinted_imag


im = cv2.imread('Photos/2.png')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
plt.figure("Grey")
plt.imshow(imgray)

ret,thresh = cv2.threshold(imgray, 150 ,255,0)
image, contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img = cv2.drawContours(im, contours, -1, (0,255,0), 3)
plt.figure("Detected")
plt.imshow(img)
plt.show()
plt.imshow(im)
plt.show()

#tinted_windmills = shade(windmills, 0.7)

#plt.imshow(windmills)