# reference: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html



import numpy as np
import cv2

from matplotlib import pyplot as plt


MIN_MATCH_COUNT = 3
MIN_KEY_POINTS = 15
MATCH_THRESHOLD = 16


def calculate_similarity_goodness(good, kp1, kp2):
    number_keypoints = len(kp1) + 1
    if number_keypoints < MIN_KEY_POINTS:
         return 100
    goodness = len(good) / number_keypoints * 100
    # print("KP1:", len(kp1), ", KP2:", len(kp2))
    # print("How good it's the match: ", goodness)
    return goodness

def draw1(img1, kp1, img2, kp2, good):
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, outImg=None, flags=2)
    calculate_similarity_goodness(good, kp1, kp2)

    plt.imshow(img3)
    plt.show()


def draw2(img1, kp1, img2, kp2, good, matches_mask):
    length = len(matches_mask) if matches_mask else 0
    good = [good[index] for index in range(0, length) if matches_mask[index]]
    if matches_mask:
        print("new_good/old_good = ", sum(matches_mask)*100/len(matches_mask))
    calculate_similarity_goodness(good, kp1, kp2)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2, singlePointColor=None)
    #img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2, singlePointColor=None)

    plt.imshow(img3)
    plt.show()


def SIFT_detector(img1 ,img2, gamma_goodness = 0.85):
    # img1 size must be not smaller than img2

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # matches = bf.match(des1, des2)
    # print("hahahaha", des1, matches)
    # Apply ratio test
    good = []

    for m,n in matches:
        # print(m.distance,n.distance)
        if m.distance < gamma_goodness*n.distance:
            good.append([m])
    # draw1(img1, kp1, img2, kp2, good)

    matches_mask, img_with_objects = find_objects(good, kp1 ,kp2, img1, img2)
    draw2(img1, kp1, img_with_objects, kp2, good, matches_mask)

    return img_with_objects




def SIFT_detector_on_segments(img1 ,img2, gamma_goodness = 0.8):
    # img1 size must be not smaller than img2

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    if not (kp1 and kp2):
        return False, 0
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    if len(np.array(matches).shape) != 2 or np.array(matches).shape[1] != 2:
        good = matches
    else:
        # Apply ratio test
        good = []
        #print("Matches", matches)
        for m, n in matches:
            # print(m.distance,n.distance)
            if m.distance < gamma_goodness*n.distance:
                good.append([m])

    return is_match(good, kp1, kp2)


def find_objects(good, kp1 , kp2, img1, img2):
    if len(good) > MIN_MATCH_COUNT:
    # if calculate_similarity_goodness(good, kp1, kp2) > 0.1:
        src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        tmp_img = img2.copy()
        img3 = cv2.polylines(tmp_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        if len(kp1) > 0.8*len(kp2):
            print("Match ")
        else:
            print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
        img3 = None
    return matchesMask, img3


def is_match(good, kp1 , kp2):
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    else:
        matchesMask = np.zeros(len(good), dtype=int)

    good = [good[index] for index in range(0, len(good)) if matchesMask[index]]
    similarity = calculate_similarity_goodness(good, kp1, kp2)
    return True if similarity > MATCH_THRESHOLD else False , similarity


if __name__ == "__main__":
    img1 = cv2.imread('Photos/duck_2.png')

    # dst = cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 7, 21)
    # img1 = dst

    img2 = cv2.imread('Photos/Pls.png', 0)  # trainImage
    img1 = cv2.imread('Photos/first_slice.png', 0)  # trainImage
    SIFT_detector(img1, img2)