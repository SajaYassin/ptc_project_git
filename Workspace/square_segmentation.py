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


def crop_image(source_image, implify_on_image=[]):
    # from: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
    # Coordinates of non-black pixels.

    (x_coord, y_coord, _) = np.where(source_image == 255)

    x0, x1 = x_coord.min(axis=0), x_coord.max(axis=0) + 1
    y0, y1 = y_coord.min(axis=0), y_coord.max(axis=0) + 1  # slices are exclusive at the top

    # Get the contents of the bounding box as an image.
    cropped = implify_on_image[x0:x1, y0:y1] if implify_on_image.any() else source_image[x0:x1, y0:y1]
    return cropped


def convert_color_scale(image, channel='Green'):
    channel_to_array = {'Blue': [0,1], 'Red': [1,2], 'Green': [0,2]}
    image[:,:, channel_to_array[channel]] = 0


def iterate_over_slices_2(old_image, new_image, output, disp_x_percentage=0.1, disp_y_percentage=0.1, epsilon=20):
    height, width = old_image.shape[:2]
    disp_x = int(disp_x_percentage*width)
    disp_y = int(disp_y_percentage*height)
    for y in range(0, height, disp_y):
        for x in range(0, width, disp_x):
            # print(x,y)
            cropped_old = old_image[y : y + disp_y, x : x + disp_x]
            cropped_new = new_image[max(0, y - epsilon) : min(height, y + disp_y + epsilon), max(0, x - epsilon) : min(width, x + disp_x + epsilon)]
            # print("In sift, old.zise:", cropped_old.shape[:2], "img2.size:", cropped_new.shape[:2])
            # print(cropped_new)
            # cv2.imshow("New", cropped_new)
            # cv2.imshow("Old", cropped_old)

            is_match, similarity = SIFT.SIFT_detector_on_segments(cropped_old, cropped_new)
            if not is_match:
                print("X:", x, "Y:", y, "similarity:", similarity)
                convert_color_scale(output[y : y + disp_y, x : x + disp_x], 'Blue')
            # cv2.imshow('Modified_Old', old_image)



def iterate_over_slices(old_image, new_image, image_slic):
    for (seg_number, segVal) in enumerate(np.unique(image_slic)):
        mask = np.zeros(old_image.shape[:2], dtype="uint8")
        mask[image_slic == segVal] = 255

        # cv2.imshow("Mask", mask)
        bitwise_img = cv2.bitwise_and(old_image, old_image, mask=mask)
        # crop_nonblack_img = crop_image(bitwise_img)
        crop_nonblack_img = bitwise_img
        '''print("BBBBBA444444444444444443333333333333")
        print("Before:", bitwise_img)

        print("After:", crop_nonblack_img)
        print("WA###A##3333333333333333333333")
        '''
        # cv2.imshow("Bitwise", bitwise_img)
        # cv2.imshow("Applied", crop_nonblack_img)
        SIFT.SIFT_detector(crop_nonblack_img, new_image)
        cv2.waitKey(0)


def SLIC_algo(image, squares=True):
    num_of_segments = 20 #number_of_segments(image)
    print("Image is sliced into {} segments:".format(num_of_segments))
    image_slic = slic_squares(image, n_segments=num_of_segments) if squares else seg.slic(image, n_segments=num_of_segments, sigma=50)
    '''image_bounded = seg.mark_boundaries(image, image_slic)
    plt.imshow(image_bounded)
    plt.show()
    '''
    return image_slic


def plot_colored_changes(img1, img2, output):
    # sliced_image = SLIC_algo(img1, False)
    # iterate_over_slices(img1, img2, sliced_image)
    iterate_over_slices_2(img1, img2, disp_x_percentage=0.08, disp_y_percentage=0.2, epsilon=int(0.05 * min(img2.shape[:2])), output=output)



if __name__ == "__main__":
    # We want to differentiate between the old image and the current status.
    # Therefore: slice the original image and iterate over the slices with the new image.
    # image = cv2.imread('Photos/Pls.png')
    image_old = cv2.imread('Photos/no_marwan.png')
    image_new = cv2.imread('Photos/marwan.png')
    res = SIFT.SIFT_detector(image_new, image_old)

    old_cropped = crop_image(res, image_old)

    colored_image = image_new.copy()
    plot_colored_changes(img1=image_new, img2=old_cropped, output=colored_image)
    #plt.imshow(colored_image), plt.title("Colored"), plt.show()


    image_old = cv2.imread('Photos/no_marwan.png')
    image_new = cv2.imread('Photos/marwan.png')
    res = SIFT.SIFT_detector(image_new, image_old)

    new_cropped = crop_image(res, image_new)
    plot_colored_changes(img1=image_old, img2=new_cropped, output=colored_image)
    plt.imshow(colored_image), plt.title("Colored"), plt.show()


