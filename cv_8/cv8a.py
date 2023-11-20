import cv2
import numpy as np
import matplotlib.pyplot as plt


def part_one():
    filename = 'pvi_cv08/pvi_cv08_spz.png'
    img = cv2.imread(filename)
    img_copy = img.copy()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                  # Img 1
    rgb_copy = rgb.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    _, binary_image = cv2.threshold(hue, 90, 255, cv2.THRESH_BINARY)  # Segmentace (...to binary)
    gray = np.float32(gray)
    #_, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)  # Apply threshold
    #_, binary_image = cv2.threshold(gray, 105, 255, cv2.THRESH_BINARY)

    kernel = np.ones((15, 15), np.uint8)

    # Define the lower and upper bounds of the white color in HSV
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 30, 255], dtype=np.uint8)
    # Create a binary mask for the white color range
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    cv2.imshow('White Mask', white_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    opened_image = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # Apply morphological operations to remove text and keep rectangle
    dilated_image = cv2.dilate(opened_image, kernel, iterations=5)      # dilate bin img
    eroded_image = cv2.erode(dilated_image, kernel, iterations=2)       # erode dilated img
    reverse_eroded_image = np.zeros_like(eroded_image)
    reverse_eroded_image[eroded_image == 0] = 255
    cv2.imshow('bin basic', reverse_eroded_image)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    # Harris corner detector
    binary_image = np.float32(reverse_eroded_image)
    dst = cv2.cornerHarris(binary_image, 2, 3, 0.04)
    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img_copy[dst > 0.01*dst.max()] = [0, 0, 255]


    """Define Figure1"""

    fig = plt.figure(figsize=(10, 7))
    rows = 2
    columns = 2
    # 1st position
    fig.add_subplot(rows, columns, 1)
    plt.imshow(rgb)
    plt.title('Orig. Im.')
    # 2nd position
    fig.add_subplot(rows, columns, 2)
    plt.imshow(white_mask, cmap='jet')
    plt.title('Bin. Im.')
    plt.colorbar()
    # 3rd position
    fig.add_subplot(rows, columns, 3)
    plt.imshow(binary_image, cmap='jet')
    plt.title('Bin. Im. - Result')
    plt.colorbar()
    # 4th position
    fig.add_subplot(rows, columns, 4)
    plt.imshow(img_copy, cmap='jet')
    plt.title('Im + Harris ')
    plt.colorbar()
    # Show
    plt.show()


if __name__ == "__main__":
    part_one()
