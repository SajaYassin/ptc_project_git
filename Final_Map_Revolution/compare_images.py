
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.segmentation as seg
import Sift_Algorithm as SIFT
import argparse
import os
from os.path import join



def convert_color_scale(image, channel='Green'):
    channel_to_array = {'Blue': [0,1], 'Red': [1,2], 'Green': [0,2]}
    image[:,:, channel_to_array[channel]] = 0


def plot_changes_dense_sift(img, mask, color_the_change=True):
    colored_img = img.copy()
    color, val_mask = ('Red', 0) if color_the_change else ('Green',255)
    convert_color_scale(colored_img, color)
    img[mask == val_mask] = colored_img[mask == val_mask]
    # for y in range(0, height, step_size):
    #     for x in range(0, width, step_size):
    #         if((x,y) in source_points):
    #             continue
    #         cropped = img[max(y - step_size, 0): min(height, y + step_size),
    #                   max(x - step_size, 0): min(width, x + step_size)]
    #         convert_color_scale(cropped, 'Red')
    #
    # # color the unchanges
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



def get_mask(shape,unchanged_pts,step_size=25):
    height = shape[0]
    width = shape[1]
    mask = np.zeros((height, width))
    # plot_changes_dense_sift(mask, unchanged_pts,step_size)
    for x,y in unchanged_pts:
        mask[max(y - step_size, 0): min(height, y + step_size),
                          max(x - step_size, 0): min(width, x + step_size)] = 255
    return mask



def run_comparsion(old_img_path, new_img_path, step_size=25 , max_iters=2000, color_the_change=True,ransac_reproj_threshold=9.0):
    image_old = cv2.imread(old_img_path)
    image_new = cv2.imread(new_img_path)
    # the old image is suppose to be  the synth contain the new image suppose to be the real image
    # this sift run is in order to find the right perspective and to in order to modify the synth
    # image to suit the new image comparision
    res, source_points = SIFT.SIFT_detector(image_new, image_old,max_iters=max_iters, is_dense=True, step_size=10,ransac_reproj_threshold=ransac_reproj_threshold)
    plt.imshow(res), plt.title("aligning lines")

    # resizing the synthetic image
    new_hight, new_width = image_new.shape[:2]
    old_cropped = crop_image(res, image_old)
    resized_image = cv2.resize(old_cropped, (new_width, new_hight))
    plt.imshow(resized_image), plt.title("resized image"), plt.show()

    # sift to find the changes
    res, source_points1 = SIFT.SIFT_detector(image_new, resized_image, step_size=step_size, ransac_reproj_threshold=ransac_reproj_threshold)
    # plt.imshow(res), plt.title("COLORED"), plt.show()

    # sift to find the changes
    res, source_points2 = SIFT.SIFT_detector(resized_image, image_new,step_size=step_size, ransac_reproj_threshold=ransac_reproj_threshold)
    # cv2.imshow("Result",res)

    #color the found changes and calc the mask
    source_points = list(set(source_points1 + source_points2))
    mask = get_mask(image_new.shape, list(np.int_(source_points)),step_size=step_size)
    colored_image = image_new.copy()
    plot_changes_dense_sift(colored_image, mask,color_the_change)
    plt.imshow(colored_image), plt.title("final result"), plt.show()
    return resized_image, colored_image, mask



def parse_arguments():
    parser = argparse.ArgumentParser(description='''
                  This programs takes two images of a rearranged scene.
                  It returns a mask (over the real_img) deciding what were the changes. 
                    ''')
    parser.add_argument('--synth_img', type=str, required=True, help='A synthetic image from the mesh.')
    parser.add_argument('--real_img', type=str, required=True, help='A real image from the same scene.')
    parser.add_argument('--max_iters', type=int, default=2000,  help='The maximal number of iterations Ransac will do, maximum 2000')
    parser.add_argument('--ransac_reproj_threshold', type=int, default=9.0,  help='Ransac reprojection error threshold, rangeR 0-10')
    parser.add_argument('--block_size_len', type=int, default=20, help="The length of the comparison squares")
    parser.add_argument('--out_dir',  type=str, required=True, help="Path to a directory to put the result in.")
    parser.add_argument('--more_details', action="store_true",
                        help="Generates the aligned synthetic image and the colored real image")
    parser.add_argument('--color_the_change', action="store_true",
                        help="Color the changed portion on the real image. Otherwise, color the unchanged.")
    args_package = parser.parse_args()
    for arg in ["synth_img", "real_img", "out_dir"]:
        if(not os.path.exists(getattr(args_package, arg))):
            parser.error("Error: {0} does not xist...".format(arg))
    return [getattr(args_package, arg) for arg in vars(args_package)]


def createDire(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Directory ", path, " Created ")
    else:
        print("Directory ", path, " already exists")




if __name__ == "__main__":
    # rel_path = "../../photos/"
    # rel_path_res = "../../res"
    # createDire(rel_path_res)
    # # run synth vs real
    # for index in range(300, 301):
    #     frame = 'frame' + str(index)
    #     frame_path = rel_path + frame
    #     print("Frame_path: ", frame_path)
    #     resized_img, colored_result, mask = run_comparsion(old_img_path=join(frame_path + 's.png'),
    #                                                        new_img_path=join(frame_path + '.jpg'))
    #     result_path = rel_path_res + frame + '_res.jpg'
    #     print("res path", result_path)
    #     cv2.imwrite(filename=result_path, img=mask)
    #     result_path = rel_path_res + frame + '_res_synth.jpg'
    #     cv2.imwrite(filename=result_path, img=resized_img)
    # exit(0)
    synth_img, real_img, max_iters, ransac_reproj_threshold, block_size_len, out_dir, more_details, color_the_change = parse_arguments()
    resized_img, colored_result, mask = run_comparsion(old_img_path=synth_img, new_img_path=real_img,
                                                       max_iters=max_iters, step_size=block_size_len, color_the_change=color_the_change, ransac_reproj_threshold=ransac_reproj_threshold)
    real_img_file_name = os.path.basename(real_img).rsplit('.', 1)[0]
    cv2.imwrite(filename=join(out_dir,real_img_file_name + "_mask_result.png"), img=mask)
    print(join(out_dir, real_img_file_name + "_mask_result.png"))
    if more_details:
        cv2.imwrite(filename=join(out_dir, real_img_file_name, "aligned_image.png"), img=resized_img)
        cv2.imwrite(filename=join(out_dir, real_img_file_name, "colored_image.png"), img=resized_img)

    exit(0)
