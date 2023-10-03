import cv2
import matplotlib.pyplot as plt
import numpy as np

im = cv2.imread('cv02_02.bmp')
#im2 = im.copy()
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.title('Image Before')
plt.imshow(im)
plt.show()
rows, cols = gray.shape


def rotate(image, angle, center=None, scale=0.50):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

#TODO: Psani do obrazku
def write_to_image(im, org, text, text_color='black'):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 0.5
    # color
    if text_color == 'black':
        colT = (0, 0, 0)  # black text
    else:
        colT = (255, 255, 255)  # white text
    # Line thickness of 2 px
    thickness = 1
    cv2.putText(im, text, org, font, fontScale, colT, thickness, cv2.LINE_AA)


colors_array_list = [
    np.array([255, 128, 200]),
    np.array([127, 0, 127]),
    np.array([255, 127, 0]),
    np.array([255, 242, 0]),
    np.array([0, 0, 255]),
    np.array([0, 255, 0]),
    np.array([255, 0, 0])]


def color_picker(image):
    height, width, channels = image.shape
    black_array = np.array([0, 0, 0])
    # Loop through all pixels
    for y in range(height):
        for x in range(width):
            # Get the color of the pixel at position (x, y)
            pixel_color = image[y, x]
            # Check if the arrays are equal
            if not np.array_equal(pixel_color, black_array):
                if np.array_equal(pixel_color, colors_array_list[0]):
                    text = 'pink'
                    return text
                elif np.array_equal(pixel_color, colors_array_list[1]):
                    text = 'purple'
                    return text
                elif np.array_equal(pixel_color, colors_array_list[2]):
                    text = 'orange'
                    return text
                elif np.array_equal(pixel_color, colors_array_list[3]):
                    text = 'yellow'
                    return text
                elif np.array_equal(pixel_color, colors_array_list[4]):
                    text = 'blue'
                    return text
                elif np.array_equal(pixel_color, colors_array_list[5]):
                    text = 'green'
                    return text
                elif np.array_equal(pixel_color, colors_array_list[6]):
                    text = 'red'
                    return text
                else: print('Color not in list of colors!')


def crop_img_part(img):
    angle_val = 45
    for x in range(50, rows, 100):
        for y in range(50, cols, 100):
            crpd_img = img[x-50:x+50, y-50:y+50]
            color_text = color_picker(crpd_img)
            if angle_val % 2 == 1:
                scale_num = 1.0
            else: scale_num = 0.5
            rotated_crpd_img = rotate(crpd_img, angle_val, center=None, scale=scale_num)
            # inkrementace uhlu a pocitadla kazde druhe sipky
            angle_val += 45
            img[x-50:x+50, y-50:y+50] = rotated_crpd_img
            write_to_image(img, (y-20, x+30), color_text, 'white')
    return img


if __name__ == "__main__":
    im2 = crop_img_part(im)
    cv2.imwrite('cv3_with_text.jpg', im2)
    plt.title('Image After')
    plt.imshow(im2)
    plt.show()
