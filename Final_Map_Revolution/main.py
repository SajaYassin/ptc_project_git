
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.segmentation as seg
import Sift_Algorithm as SIFT
from Sift_Algorithm import step_size
from os.path import join


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


def crop_2(image):
    image_copy = image.copy()

    black_pixels_mask = np.all(image == [255, 0, 0], axis=-1)

    # non_black_pixels_mask = np.any(image != [0, 0, 0], axis=-1)
    non_black_pixels_mask = ~black_pixels_mask

    image_copy[black_pixels_mask] = [255, 255, 255]
    image_copy[non_black_pixels_mask] = [0, 0, 0]
    return image_copy

    # plt.imshow(image_copy)
    # plt.show()


def crop_image(source_image, implify_on_image=[]):
    # from: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
    # Coordinates of non-black pixels.
    (x_coord, y_coord) = np.where(np.all(source_image == (255, 0, 0), axis=-1))

    x0, x1 = x_coord.min(axis=0), x_coord.max(axis=0) + 1
    y0, y1 = y_coord.min(axis=0), y_coord.max(axis=0) + 1  # slices are exclusive at the top

    # Get the contents of the bounding box as an image.
    cropped = implify_on_image[x0:x1, y0:y1] if implify_on_image.any() else source_image[x0:x1, y0:y1]
    return cropped


def my_main(old_img_path, new_img_path):
    image_old = cv2.imread(old_img_path)

    image_new = cv2.imread(new_img_path)
    new_hight, new_width = image_new.shape[:2]
    res, source_points = SIFT.SIFT_detector(image_new, image_old, is_dense=False)
    # plt.imshow(res), plt.title("res"), plt.show()

    # resizing the synthetic image
    old_cropped = crop_image(res, image_old)
    resized_image = cv2.resize(old_cropped, (new_width, new_hight))
    res, source_points = SIFT.SIFT_detector(image_new, resized_image)
    # plt.imshow(res), plt.title("COLORED"), plt.show()
    colored_image = image_new.copy()
    plot_changes_dense_sift(colored_image, source_points)
    # plt.imshow(colored_image), plt.title("Colored1"), plt.show()


    res, source_points = SIFT.SIFT_detector(resized_image, image_new)
    # cv2.imshow("Result",res)

    # new_cropped = crop_image(res, image_new)
    # plot_colored_changes(img1=image_old, img2=new_cropped, output=colored_image)
    plot_changes_dense_sift(colored_image, source_points)
    # plt.imshow(colored_image), plt.title("Colored2"), plt.show()
    return resized_image, colored_image







if __name__ == "__main__":
    rel_path = "../Workspace/Photos"
    rel_interesting =  "../../interesting_images"
    rel_mor = "../../images from Mor/"
    # We want to differentiate between the old image and the current status.
    # Therefore: slice the original image and iterate over the slices with the new image.
    # image = cv2.imread('Photos/Possible.jpg')
    # my_main(old_img_path='Photos/empty_mech.png', new_img_path='Photos/marwan_mech.png')
    # my_main(old_img_path='Photos/empty_mech.png', new_img_path='Photos/saja_mech.png')
    # my_main(old_img_path=rel_mor+'/frame36.jpg', new_img_path=rel_mor+'/frame36s.png')
    # my_main(old_img_path=rel_mor+'/frame195s.png', new_img_path=rel_mor+'/frame195.jpg')
    # res = my_main(old_img_path=rel_interesting+'/computers_per4.jpg', new_img_path=rel_interesting+'/computers_per3.jpg')
    # plt.imshow(res), plt.title("Final Image"), plt.show()

    for index in range(202, 203):
        frame = 'frame' + str(index)
        frame_path = join(rel_mor, 'real_synth',frame)
        print("Frame_path: ", frame_path)
        resized_img, colored_result = my_main(old_img_path=join(frame_path + 's.png'), new_img_path=join(frame_path + '.jpg'))
        result_path = join(rel_mor, 'real_synth_results_new', frame+'_res.jpg')
        cv2.imwrite(filename=result_path,img=colored_result)
        result_path = join(rel_mor, 'real_synth_results_new', frame + '_res_synth.jpg')
        cv2.imwrite(filename=result_path,img=resized_img)


    # # my_main(old_img_path=rel_path+'/office_real.png', new_img_path=rel_path+'/office_synth.png')
    # my_main(old_img_path='Photos/Mor/frame1.jpg', new_img_path='Photos/Mor/frame1s.png')

    # my_main(old_img_path='Photos/Mor/frame1s.png', new_img_path='Photos/Mor/frame1.jpg')