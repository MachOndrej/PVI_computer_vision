import cv2


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


def color_of_square(image_file, new_image_name):
    # load img
    im = cv2.imread(image_file)
    im2 = im.copy()  # copy for work
    # convert to grayscale and HSV
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    rows, cols = gray.shape
    color_list = ('red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'black', 'white')

    # Loop through each pixel
    for i in range(100, rows, 200):
        for j in range(100, cols, 200):
            # kontrola black/white podle grayscale image
            if gray[i, j] == 0:
                write_to_image(im2, (j - 30, i), color_list[7], 'white')
                continue
            if gray[i, j] == 255:
                write_to_image(im2, (j - 30, i), color_list[8])
                continue
            # kontrola zbytku barev podle HSV image
            hue_val = hsv[i, j, 0]
            if 0 <= hue_val < 10:
                write_to_image(im2, (j - 30, i), color_list[0])
            elif 10 <= hue_val < 23:
                write_to_image(im2, (j - 30, i), color_list[1])
            elif 23 <= hue_val < 40:
                write_to_image(im2, (j - 30, i), color_list[2])
            elif 40 <= hue_val < 80:
                write_to_image(im2, (j - 30, i), color_list[3])
            elif 80 <= hue_val < 140:
                write_to_image(im2, (j - 30, i), color_list[4])
            elif 140 <= hue_val < 155:
                write_to_image(im2, (j - 30, i), color_list[5])
            else:
                write_to_image(im2, (j, i), color_list[6])

    # zapis do noveho obrazku
    cv2.imwrite(new_image_name, im2)
    #cv2.imshow('with text', im2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    color_of_square('cv02_01.bmp', 'cv02_with_text.jpg')
    print('done')
