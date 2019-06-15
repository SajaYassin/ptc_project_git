
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.segmentation as seg
import Sift_Algorithm as SIFT
import os
from os.path import join



def convert_color_scale(image, channel='Green'):
    channel_to_array = {'Blue': [0,1], 'Red': [1,2], 'Green': [0,2]}
    image[:,:, channel_to_array[channel]] = 0


def plot_changes_dense_sift(img, source_points, step_size):
    height, width = img.shape[:2]
    for y in range(0, height, step_size*2):
        for x in range(0, width, step_size*2):
            if((x,y) in source_points):
                continue
            cropped = img[max(y - step_size, 0): min(height, y + step_size),
                      max(x - step_size, 0): min(width, x + step_size)]
            convert_color_scale(cropped, 'Red')

    # color the unchanges
    # for x, y in source_points:
    #     x, y = int(x), int(y)
    #     cropped = img[max(y - step_size, 0): min(height, y + step_size),
    #                   max(x - step_size, 0): min(width, x + step_size)]
    #     convert_color_scale(cropped, 'Green')


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
    # the old image suppose the synth contain the new image suppose to be the real image
    # this sift run is in order to find the right perspective and to in order to modify the synth
    # image to suit the new image comparision
    res, source_points = SIFT.SIFT_detector(image_new, image_old, is_dense=True, step_size=10)
    # plt.imshow(res), plt.title("res"), plt.show()

    # resizing the synthetic image
    new_hight, new_width = image_new.shape[:2]
    old_cropped = crop_image(res, image_old)
    resized_image = cv2.resize(old_cropped, (new_width, new_hight))
    # plt.imshow(resized_image), plt.title("resized_image"), plt.show()

    # sift to find the changes
    res, source_points = SIFT.SIFT_detector(image_new, resized_image, step_size=25)
    # plt.imshow(res), plt.title("COLORED"), plt.show()

    # color the found changes on the result image
    colored_image = image_new.copy()
    plot_changes_dense_sift(colored_image, source_points, step_size=25)
    # plt.imshow(colored_image), plt.title("Colored1"), plt.show()
    # sift to find the changes
    res, source_points = SIFT.SIFT_detector(resized_image, image_new)
    # cv2.imshow("Result",res)

    # new_cropped = crop_image(res, image_new)
    # plot_colored_changes(img1=image_old, img2=new_cropped, output=colored_image)
    plot_changes_dense_sift(colored_image, source_points, step_size=25)
    plt.imshow(colored_image), plt.title("Colored2"), plt.show()
    return resized_image, colored_image


def parking_diff(old_img_path, new_img_path, step):
    image_old = cv2.imread(old_img_path)
    image_new = cv2.imread(new_img_path)

    # plt.imshow(image_new), plt.title("COLORED"), plt.show()

    # no need to resize in this case, assuming the images are in the same size
    # sift to find the changes
    new_width,new_hight = (640,360)
    image_old = cv2.resize(image_old, (new_width, new_hight))
    image_new = cv2.resize(image_new, (new_width, new_hight))
    res, source_points1 = SIFT.SIFT_detector(image_new, image_old, step_size=step)
    # plt.imshow(res), plt.title("COLORED"), plt.show()

    # color the found changes on the result image
    # plt.imshow(colored_image), plt.title("Colored1"), plt.show()
    # sift to find the changes
    res, source_points2 = SIFT.SIFT_detector(image_old, image_new, step_size=step)
    # cv2.imshow("Result",res)

    # new_cropped = crop_image(res, image_new)
    # plot_colored_changes(img1=image_old, img2=new_cropped, output=colored_image)
    source_points = list(set(source_points1 + source_points2))
    colored_image = image_new.copy()
    plot_changes_dense_sift(colored_image, source_points, step_size=step)
    # plt.imshow(colored_image), plt.title("Colored2"), plt.show()
    return  colored_image


def createDire(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Directory ", path, " Created ")
    else:
        print("Directory ", path, " already exists")


def run_parking_test(collection_name, step):
    collection_num = str(collection_name)
    path_first_vs_all = join("../../parking_res", "step" + str(step), collection_num, "first vs all")
    path_one_vs_next = join("../../parking_res", "step" + str(step), collection_num, "one vs nxt")
    src_path = join("../../PTC photos", collection_num + "GOPRO")
    images_name = os.listdir(src_path)
    first_file = images_name[0]

    createDire(path_first_vs_all)
    createDire(path_one_vs_next)

    for index in range(0, len(images_name) - 1):
        first_path= join(src_path, first_file)
        old_path = join(src_path, images_name[index])
        next_path = join(src_path, images_name[index + 1])
        print("testing ", images_name[index]," in ",collection_num, " step ",str(step),)
        colored_result = parking_diff(old_img_path=old_path ,
                                              new_img_path=next_path, step=step)
        result_path = join(path_one_vs_next, 'res_' + images_name[index])
        cv2.imwrite(filename=result_path, img=colored_result)

        colored_result = parking_diff(old_img_path=first_path,
                                              new_img_path=old_path, step=step)
        result_path = join(path_first_vs_all, 'res_' + images_name[index])
        cv2.imwrite(filename=result_path, img=colored_result)



if __name__ == "__main__":
    # rel_path = "../../PTC photos/100GOPRO/"
    # rel_path_res = "../../parking_res/101G/step20 one vs first"

    # rel_interesting = "../../interesting_images"
    # rel_mor = "../../images from Mor/"
    # We want to differentiate between the old image and the current status.
    # Therefore: slice the original image and iterate over the slices with the new image.
    # image = cv2.imread('Photos/Possible.jpg')
    # my_main(old_img_path='Photos/empty_mech.png', new_img_path='Photos/marwan_mech.png')
    # my_main(old_img_path='Photos/empty_mech.png', new_img_path='Photos/saja_mech.png')
    # my_main(old_img_path=rel_mor+'/frame36.jpg', new_img_path=rel_mor+'/frame36s.png')
    # my_main(old_img_path=rel_mor+'/frame195s.png', new_img_path=rel_mor+'/frame195.jpg')

    # resized_image, res = my_main(rel_path + "zoom_in2.jpg",rel_path + "zoom_in1.jpg" )
    # plt.imshow(res), plt.title("Final Image"), plt.show()
    # cv2.imwrite(filename=rel_path_res, img=res)
    # exit(0)


    ''' 
        run synth vs real
        for index in range(1, 10):
        frame = 'frame' + str(index)
        frame_path = rel_path + frame
        print("Frame_path: ", frame_path)
        resized_img, colored_result = my_main(old_img_path=join(frame_path + 's.png'), new_img_path=join(frame_path + '.jpg'))
        result_path = rel_path_res + frame + '_res.jpg'
        print("res path", result_path)
        cv2.imwrite(filename=result_path,img=colored_result)
        result_path = rel_path_res + frame + '_res_synth.jpg'
        cv2.imwrite(filename=result_path,img=resized_img)
    '''
    for step_size in [20, 25, 30]:
        for collection_num in range(100,103):
            run_parking_test(collection_num, step=step_size)

