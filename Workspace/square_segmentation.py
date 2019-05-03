import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.segmentation as seg
import using_Sift as SIFT


def slic_squares(image, n_segments=100):
    squares_array = np.zeros(image.shape[:2], dtype="uint8")
    return seg.slic(squares_array, n_segments=n_segments)




def number_of_segments(image):
    return image.size//(10**(len(str(image.size))*0.8))





def print_np_array(array):
    num = 0
    for line in array:
        num += 1
        print(num, line)


def crop_image(image):
    # from: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
    # Coordinates of non-black pixels.
    (x_coord, y_coord, _) = np.where(image > 0)
    x0, x1 = x_coord.min(axis=0), x_coord.max(axis=0) + 1
    y0, y1 = y_coord.min(axis=0), y_coord.max(axis=0) + 1  # slices are exclusive at the top

    # Get the contents of the bounding box as an image.
    cropped = image[x0:x1, y0:y1]
    return cropped


def iterate_over_slices_2(old_image, new_image, disp_x_percentage = 0.1, disp_y_percentage=0.1, epsilon=5):
    height, width = old_image.shape[:2]
    disp_x = int(disp_x_percentage*width)
    disp_y = int(disp_y_percentage*height)
    for y in range(0, height, disp_y):
        for x in range(0, width, disp_x):
            cropped_old = old_image[y : y + disp_y, x : x + disp_x]
            cropped_new = new_image[max(0, y - epsilon) : min(height, y + disp_y + epsilon), max(0, x - epsilon) : min(width, x + disp_x + epsilon)]
            cv2.imshow("New", cropped_new)
            cv2.imshow("Old", cropped_old)
            SIFT.SIFT_detector(cropped_old, cropped_new)
            cv2.waitKey(0)


def iterate_over_slices(old_image, new_image, image_slic):
    for (seg_number, segVal) in enumerate(np.unique(image_slic)):
        mask = np.zeros(old_image.shape[:2], dtype="uint8")
        mask[image_slic == segVal] = 255

        cv2.imshow("Mask", mask)
        bitwise_img = cv2.bitwise_and(old_image, old_image, mask=mask)
        crop_nonblack_img = crop_image(bitwise_img)
        '''print("BBBBBA444444444444444443333333333333")
        print("Before:", bitwise_img)

        print("After:", crop_nonblack_img)
        print("WA###A##3333333333333333333333")
        '''
        cv2.imshow("Bitwise", bitwise_img)
        cv2.imshow("Applied", crop_nonblack_img)
        SIFT.SIFT_detector(crop_nonblack_img, new_image)
        cv2.waitKey(0)


def SLIC_algo(image, squares=True):
    num_of_segments = number_of_segments(image)
    print("Image is sliced into {} segments:".format(num_of_segments))
    image_slic = slic_squares(image, n_segments=num_of_segments) if squares else seg.slic(image, n_segments=num_of_segments, sigma=50)
    '''image_bounded = seg.mark_boundaries(image, image_slic)
    plt.imshow(image_bounded)
    plt.show()
    '''
    return image_slic


if __name__ == "__main__":
    # We want to differentiate between the old image and the current status.
    # Therefore: slice the original image and iterate over the slices with the new image.
    # image = cv2.imread('Photos/Pls.png')
    image_old = cv2.imread('Photos/ducks_around_computer_3.png')
    image_new = cv2.imread('Photos/ducks_around_computer_2.png')

    # sliced_image = SLIC_algo(image_old)
    iterate_over_slices_2(image_new, image_old, disp_x_percentage=0.2, disp_y_percentage=0.1, epsilon=int(0.05* min(image_old.shape[:2])))

