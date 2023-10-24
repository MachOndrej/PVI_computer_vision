import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label
# Load file and convert to gray, hue
img = cv2.imread('pvi_cv05_mince_noise.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue = hsv[:, :, 0]


# image_list = [img1, hist1, spec1, img2, hist2, spec2 ]
def visualization(image_list, first_img_name, second_img_name, title):
    # Start suobplot
    fig, axs = plt.subplots(2, 3, constrained_layout=True, figsize=(10, 7))
    # Set up for visualiyation - 1st row    (axs[row,col])
    axs[0, 0].imshow(image_list[0], cmap='gray')
    axs[0, 0].set_title(first_img_name, fontsize=10.0)
    axs[0, 1].plot(image_list[1])
    axs[0, 1].set_title('Gray Image Histogram', fontsize=10.0)
    axs[0, 2].set_title('Bin Image from Gray', fontsize=10.0)
    # 2nd row
    axs[1, 0].imshow(image_list[3], cmap='jet')
    axs[1, 0].set_title(second_img_name, fontsize=10.0)
    axs[1, 1].plot(image_list[4])
    axs[1, 1].set_title('Hue Image Histogram', fontsize=10.0)
    axs[1, 2].set_title('Bin Image from Hue', fontsize=10.0)
    # Set up colorbars
    fig.colorbar(axs[0, 2].imshow(image_list[2], cmap='jet'), ax=axs[0, 2], orientation='vertical')
    fig.colorbar(axs[1, 2].imshow(image_list[5], cmap='jet'), ax=axs[1, 2], orientation='vertical')
    # Set up title
    fig.suptitle(title, fontsize=10.0)
    plt.show()


def create_binary_image(original_image, threshold_value):
    if original_image is None:
        raise ValueError("Image not found. Please provide a valid image path.")
    # Apply a threshold to create the binary image
    _, binary_image = cv2.threshold(original_image, threshold_value, 1, cv2.THRESH_BINARY)

    return binary_image


def main():
    # 1)
    gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    gray_bin_img = create_binary_image(gray, 130)
    hue_hist = cv2.calcHist([hue], [0], None, [256], [0, 256])
    hue_bin_img = create_binary_image(hue, 40)
    hue_bin_img = np.ones_like(hue_bin_img) - hue_bin_img
    gray_hue_list = [gray, gray_hist, gray_bin_img, hue, hue_hist, hue_bin_img]
    visualization(gray_hue_list, 'Gray Image', 'Hue Image', ' ')
    # 2)
    # Define Figure1
    fig = plt.figure(figsize=(10, 7))
    rows = 1
    columns = 2
    # 1st position
    fig.add_subplot(rows, columns, 1)
    plt.imshow(hue_bin_img, cmap='jet')
    plt.title('Binary Image from Hue')
    plt.colorbar()
    # 2nd position
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening_hue = cv2.morphologyEx(hue_bin_img, cv2.MORPH_OPEN, kernel)
    fig.add_subplot(rows, columns, 2)
    plt.imshow(opening_hue, cmap='jet')
    plt.title('Binary Image from Hue - Opening')
    plt.colorbar()
    # Show first pair
    plt.show()
    # Define Figure2
    fig = plt.figure(figsize=(10, 7))
    rows = 1
    columns = 2
    # 1st position
    fig.add_subplot(rows, columns, 1)
    plt.imshow(opening_hue, cmap='jet')
    plt.title('Binary Image from Hue - Opening')
    plt.colorbar()
    # 2nd position
    label_hue, ncc = label(opening_hue)
    fig.add_subplot(rows, columns, 2)
    plt.imshow(label_hue, cmap='jet')
    plt.title('Binary Image from Hue - Label')
    plt.colorbar()
    plt.show()
    # 3)
    img_org_copy = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    img_org_classification = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

    classification_vals = [4000, 4500, 5300, 6500]
    classification_dic = {
        4000: '1 Kc',
        4500: '2 Kc',
        5300: '5 Kc',
        6500: '10 Kc'
    }

    area_center_dic = {}

    for i in range(1, ncc+1):
        marked_area = np.where(label_hue == i)
        area_size = len(marked_area[0])
        area_center = (int(np.mean(marked_area[1])), int(np.mean(marked_area[0])))
        area_center_dic[i] = area_center
        cv2.putText(img_org_copy, str(area_size), area_center, cv2.FONT_HERSHEY_SIMPLEX, 1.0, [0, 0, 255])

        for val in classification_vals:
            if area_size < val:
                cv2.putText(img_org_classification, classification_dic[val], area_center,
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, [0, 0, 255])
                break
    # Img with values
    plt.figure()
    plt.imshow(img_org_copy)
    plt.show(block=False)
    # Img with Kc
    plt.figure()
    plt.imshow(img_org_classification)
    plt.show(block=False)
    plt.show()


if __name__ == "__main__":
    main()
