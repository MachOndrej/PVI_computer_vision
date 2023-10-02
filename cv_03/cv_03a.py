import cv2
import matplotlib.pyplot as plt

im = cv2.imread('cv02_02.bmp')

im2 = im.copy()
hsv = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

rows, cols = gray.shape


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def write_to_image(im, org, text, text_color='black'):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 1
    # color
    if text_color == 'black':
        colT = (0, 0, 0)  # black text
    else:
        colT = (255, 255, 255)  # white text
    # Line thickness of 2 px
    thickness = 2
    cv2.putText(im, text, org, font, fontScale, colT, thickness, cv2.LINE_AA)


def crop_img_part(img):
    part_num = 0
    angle_val = 45
    for x in range(50, rows, 100):
        for y in range(50, cols, 100):
            crpd_img = img[x-50:x+50, y-50:y+50]
            rotated_crpd_img = rotate(crpd_img, angle_val)
            angle_val += 45
            img[x-50:x+50, y-50:y+50] = rotated_crpd_img
            plt.title('IMage Cropped' + str(part_num))
            plt.imshow(rotated_crpd_img)
            plt.show()


crop_img_part(im)
plt.title('IMage ')
plt.imshow(im)
plt.show()
