from __future__ import print_function
import numpy as np
import cv2 as cv
import argparse
from matplotlib import pyplot as plt


def first_method():
    MIN_MATCH_COUNT = 10
    img1 = cv.imread('pvi_cv09/obcansky_prukaz_cr_sablona_2012_2014.png', cv.IMREAD_GRAYSCALE)      # queryImage
    img2 = cv.imread('pvi_cv09/CA10_01.jpg', cv.IMREAD_GRAYSCALE)                                   # trainImage

    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    # TODO: descriptory a zajmove body here -> najit kde kreslim okraj
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([ [0, 0], [0, h-1], [w-1, h-1], [w-1, 0] ]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)

        img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),      # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,     # draw only inliers
                       flags=2)

    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    plt.imshow(img3, 'gray')
    plt.show()


def second_method():
    parser = argparse.ArgumentParser(description='Code for Feature Matching with FLANN tutorial.')
    parser.add_argument('--input1', help='Path to input image 1.', default='box.png')
    parser.add_argument('--input2', help='Path to input image 2.', default='box_in_scene.png')
    args = parser.parse_args()

    img_object = cv.imread('pvi_cv09/obcansky_prukaz_cr_sablona_2012_2014.png', cv.IMREAD_GRAYSCALE)
    img_scene = cv.imread('pvi_cv09/TA10_01.jpg', cv.IMREAD_GRAYSCALE)

    if img_object is None or img_scene is None:
        print('Could not open or find the images!')
        exit(0)

    # -- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    minHessian = 400
    detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)
    keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)

    # -- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)

    # -- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.75
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # -- Draw matches
    img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1] + img_scene.shape[1], 3),
                           dtype=np.uint8)

    cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, good_matches, img_matches,
                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # -- Localize the object
    obj = np.empty((len(good_matches), 2), dtype=np.float32)
    scene = np.empty((len(good_matches), 2), dtype=np.float32)
    for i in range(len(good_matches)):
        # -- Get the keypoints from the good matches
        obj[i, 0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
        obj[i, 1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
        scene[i, 0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
        scene[i, 1] = keypoints_scene[good_matches[i].trainIdx].pt[1]

    H, _ = cv.findHomography(obj, scene, cv.RANSAC)

    # -- Get the corners from the image_1 ( the object to be "detected" )
    obj_corners = np.empty((4, 1, 2), dtype=np.float32)
    obj_corners[0, 0, 0] = 0
    obj_corners[0, 0, 1] = 0
    obj_corners[1, 0, 0] = img_object.shape[1]
    obj_corners[1, 0, 1] = 0
    obj_corners[2, 0, 0] = img_object.shape[1]
    obj_corners[2, 0, 1] = img_object.shape[0]
    obj_corners[3, 0, 0] = 0
    obj_corners[3, 0, 1] = img_object.shape[0]

    scene_corners = cv.perspectiveTransform(obj_corners, H)

    # -- Draw lines between the corners (the mapped object in the scene - image_2 )
    cv.line(img_matches, (int(scene_corners[0, 0, 0] + img_object.shape[1]), int(scene_corners[0, 0, 1])), \
            (int(scene_corners[1, 0, 0] + img_object.shape[1]), int(scene_corners[1, 0, 1])), (0, 255, 0), 4)
    cv.line(img_matches, (int(scene_corners[1, 0, 0] + img_object.shape[1]), int(scene_corners[1, 0, 1])), \
            (int(scene_corners[2, 0, 0] + img_object.shape[1]), int(scene_corners[2, 0, 1])), (0, 255, 0), 4)
    cv.line(img_matches, (int(scene_corners[2, 0, 0] + img_object.shape[1]), int(scene_corners[2, 0, 1])), \
            (int(scene_corners[3, 0, 0] + img_object.shape[1]), int(scene_corners[3, 0, 1])), (0, 255, 0), 4)
    cv.line(img_matches, (int(scene_corners[3, 0, 0] + img_object.shape[1]), int(scene_corners[3, 0, 1])), \
            (int(scene_corners[0, 0, 0] + img_object.shape[1]), int(scene_corners[0, 0, 1])), (0, 255, 0), 4)

    # -- Show detected matches
    cv.imshow('Good Matches & Object detection', img_matches)
    cv.waitKey()


def main():
    first_method()


if __name__ == "__main__":
    main()
