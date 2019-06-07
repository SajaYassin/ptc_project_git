
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.segmentation as seg
import Sift_Algorithm as SIFT
from Sift_Algorithm import step_size




def convert_color_scale(image, channel='Green'):
    channel_to_array = {'Blue': [0,1], 'Red': [1,2], 'Green': [0,2]}
    image[:,:, channel_to_array[channel]] = 0





def plot_changes_dense_sift(img, source_points):
    height, width = img.shape[:2]
    '''for y in range(0, height, step_size*2):
        for x in range(0, width, step_size*2):
            if()
    '''
    for x, y in source_points:
        x, y = int(x), int(y)
        cropped = img[max(y - step_size, 0): min(height, y + step_size),
                      max(x - step_size, 0): min(width, x + step_size)]
        convert_color_scale(cropped, 'Green')



def my_main(old_img_path, new_img_path):
    image_old = cv2.imread(old_img_path)
    image_new = cv2.imread(new_img_path)
    res, source_points = SIFT.SIFT_detector(image_new, image_old)

    # old_cropped = crop_image(res, image_old)

    colored_image = image_new.copy()
    plot_changes_dense_sift(colored_image, source_points)
    plt.imshow(colored_image), plt.title("Colored1"), plt.show()

    # plot_colored_changes(img1=image_new, img2=old_cropped, output=colored_image)
    # plt.imshow(colored_image), plt.title("Colored"), plt.show()

    image_old = cv2.imread(old_img_path)
    image_new = cv2.imread(new_img_path)
    res, source_points = SIFT.SIFT_detector(image_old, image_new)

    # new_cropped = crop_image(res, image_new)
    # plot_colored_changes(img1=image_old, img2=new_cropped, output=colored_image)
    plot_changes_dense_sift(colored_image, source_points)
    plt.imshow(colored_image), plt.title("Colored2"), plt.show()






if __name__ == "__main__":
    rel_path = "../Workspace/Photos"
    # We want to differentiate between the old image and the current status.
    # Therefore: slice the original image and iterate over the slices with the new image.
    # image = cv2.imread('Photos/Possible.jpg')
    # my_main(old_img_path='Photos/empty_mech.png', new_img_path='Photos/marwan_mech.png')
    # my_main(old_img_path='Photos/empty_mech.png', new_img_path='Photos/saja_mech.png')
    # my_main(old_img_path='Photos/no_marwan.png', new_img_path='Photos/marwan.png')
    my_main(old_img_path=rel_path+'/office_real.png', new_img_path=rel_path+'/office_synth.png')
    # my_main(old_img_path='Photos/Mor/frame1.jpg', new_img_path='Photos/Mor/frame1s.png')

    # my_main(old_img_path='Photos/Mor/frame1s.png', new_img_path='Photos/Mor/frame1.jpg')