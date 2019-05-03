import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
import using_Sift as SIFT

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image)
    ax.axis('off')
    return fig, ax


def SLIC_algo():
    image = cv2.imread('Photos/Pls.png')
    text_threshold = image > 100

    image_slic = seg.slic(image, n_segments=100, sigma=50)
    image_bounded = seg.mark_boundaries(image,image_slic)
    # seg.find_boundaries()
    plt.imshow(image_bounded)
    plt.show()

    for (i, segVal) in enumerate(np.unique(image_slic)):
        mask = np.zeros(image.shape[:2], dtype="uint8")

        mask[image_slic == segVal] = 255
        # print(mask)

        cv2.imshow("Mask", mask)
        bitwise_img = cv2.bitwise_and(image, image, mask=mask)


        cv2.imshow("Applied", bitwise_img)
        SIFT.SIFT_detector(image, bitwise_img)
        cv2.waitKey(0)





if __name__ == "__main__":
    SLIC_algo()
    # print(text, text_threshold)
    # image_show(image_slic)

    # image_show(text_threshold)

