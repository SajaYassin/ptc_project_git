# reference: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html



import numpy as np
import cv2

from matplotlib import pyplot as plt


MIN_MATCH_COUNT = 1
MIN_KEY_POINTS = 4
MATCH_THRESHOLD = 15
step_size = 25

def draw_matches(img1, kp1, img2, kp2, good):
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2, singlePointColor=None)
    plt.imshow(img3, 'gray')
    plt.show()


def calculate_similarity_goodness(good, kp1, kp2):
    number_keypoints = len(kp1) + 1
    if number_keypoints < MIN_KEY_POINTS:
         return 100
    goodness = len(good) / number_keypoints * 100
    # print("KP1:", len(kp1), ", KP2:", len(kp2))
    # print("How good it's the match: ", goodness)
    return goodness




def dense_sift(image):
    # img = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size)
          for x in range(0, gray.shape[1], step_size)]

    #img = cv2.drawKeypoints(gray, kp, img)

    #plt.figure(figsize=(20, 10))
    #plt.imshow(img)
    #plt.show()

    kp, des = sift.compute(gray, kp)
    return kp, des


def SIFT_detector(img1 ,img2, gamma_goodness=0.85 ,is_dense=True):
    # img1 size must be not smaller than img2
    # find the keypoints and descriptors with SIFT
    if(is_dense):
        kp1, des1 = dense_sift(img1)
        kp2, des2 = dense_sift(img2)
    else:
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # cv2.imshow("Keypoints1", cv2.drawKeypoints(img1, kp1, None))
    # cv2.imshow("Keypoints2", cv2.drawKeypoints(img2, kp2, None))

    # Apply ratio test that filters "bad" matches
    good = []

    for m,n in matches:
        if m.distance < gamma_goodness*n.distance:
            good.append([m])

    matches_mask, img_with_polylines = ransace_perspective(good, kp1 ,kp2, img1, img2)

    length = len(matches_mask) if matches_mask else 0
    masked_good = [good[index] for index in range(0, length) if matches_mask[index]]
    # draw_matches(img1, kp1, img_with_polylines, kp2, masked_good)

    src_pts = [kp1[m[0].queryIdx].pt for m in masked_good]

    return img_with_polylines, src_pts


def ransace_perspective(good, kp1 , kp2, img1, img2):
    '''

    :param good:
    :param kp1:
    :param kp2:
    :param img1: source image
    :param img2: image to compare with the source image
    :return: img3 = img1 with red polylines that describes the perspective of img2 in img1
    '''

    if len(good) > MIN_MATCH_COUNT:
    # if calculate_similarity_goodness(good, kp1, kp2) > 0.1:
        src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 9.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(pts, M)

        # img_with_perspective = img1.copy()
        # # plt.imshow(img_with_perspective), plt.title("Before applying Perspective"), plt.show()
        # cv2.imshow("Before applying Perspective", img_with_perspective)
        # cv2.warpPerspective(img1, M, dsize=img1.shape[:2], dst=img_with_perspective)
        # # plt.imshow(img_with_perspective), plt.title("AFter applying Perspective"), plt.show()
        # cv2.imshow("AFter applying Perspective", img_with_perspective)
        # cv2.imshow("AFter applying Perspective img1", img1)
        tmp_img = img2.copy()
        img3 = cv2.polylines(tmp_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        # img3 = cv2.polylines(img_with_perspective, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        # plt.imshow(img3), plt.title("Img3"),  plt.show()
        # plt.imshow(img_with_perspective), plt.title("Perspective"), plt.show()

    else:
        if len(kp1) > 0.8*len(kp2):
            print("Match ")
        else:
            print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
        img3 = None
    return matchesMask, img3


