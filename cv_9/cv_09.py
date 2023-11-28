from __future__ import print_function
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import transform
import easyocr


def cut_out_rectangle(img, dst):
    mask = np.zeros_like(img, dtype=np.uint8)
    cv.fillPoly(mask, [np.int32(dst)], 255)
    result = cv.bitwise_and(img, mask)
    return result


def cutout_face_name_surname(img):
    face = img[140:420, 15:230]
    name = img[108:132, 150:350]
    surename = img[82:110, 150:350]
    return face, name, surename


def first_method():
    MIN_MATCH_COUNT = 10
    img1 = cv.imread('pvi_cv09/obcansky_prukaz_cr_sablona_2012_2014.png', cv.IMREAD_GRAYSCALE)      # queryImage
    img2 = cv.imread('pvi_cv09/HA10_06.jpg', cv.IMREAD_GRAYSCALE)                                   # trainImage
    img_positioned = img2.copy()

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
    # Img 1 to show
    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    plt.imshow(img3)
    plt.title('1')
    plt.show()

    # Img 2 to show
    roi = cut_out_rectangle(img2, dst)
    plt.imshow(roi, 'gray')
    plt.title('2')
    plt.show()

    dst = np.array([np.int32(dst)[0][0], np.int32(dst)[1][0], np.int32(dst)[2][0], np.int32(dst)[3][0]])
    src = np.array([[0, 0], [0, 420], [669, 420], [669, 0]])

    tform3 = transform.ProjectiveTransform()
    tform3.estimate(src, dst)
    warped = transform.warp(img_positioned, tform3, output_shape=(420, 669))    # Detail of cut out
    # Img 3 to show
    plt.imshow(warped, cmap=plt.cm.gray)
    plt.title('zarovnany prukaz')
    plt.show()
    # create face, name, surname
    face, name, surname = cutout_face_name_surname(warped)
    """Define Figure3"""
    fig = plt.figure(figsize=(10, 7))
    rows = 1
    columns = 3
    # 1st position
    fig.add_subplot(rows, columns, 1)
    plt.imshow(face, 'gray')
    plt.title('foto')
    # 2nd position
    fig.add_subplot(rows, columns, 2)
    plt.imshow(surname, 'gray')
    plt.title('prijmeni')
    # 3rd position
    fig.add_subplot(rows, columns, 3)
    plt.imshow(name, 'gray')
    plt.title('jmeno')
    plt.show()

    reader = easyocr.Reader(['cs'], gpu=False)
    scaled_name = (name * 255).astype('uint8')
    text_name = reader.readtext(scaled_name, detail=0)      # .lower()
    text_name = 'jmeno: ' + text_name[0].lower()
    scaled_surname = (surname * 255).astype('uint8')
    text_surname = reader.readtext(scaled_surname, detail=0)
    text_surname = 'prijmeni: ' + text_surname[0].lower()
    face = cv.putText(img=face,
                      text=text_name,
                      org=(10, 15),
                      fontFace=cv.FONT_HERSHEY_SIMPLEX,
                      fontScale=0.4,
                      color=(0, 0, 255),
                      thickness=1)
    face = cv.putText(img=face,
                      text=text_surname,
                      org=(10, 30),
                      fontFace=cv.FONT_HERSHEY_SIMPLEX,
                      fontScale=0.4,
                      color=(0, 0, 255),
                      thickness=1)
    print(text_name)
    print(text_surname)
    #cv.imwrite('cv9_result.jpg', face)
    plt.imshow(face, cmap='gray')
    plt.title('final')
    plt.show()


def main():
    first_method()


if __name__ == "__main__":
    main()
