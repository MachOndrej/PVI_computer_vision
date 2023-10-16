import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def create_image_list(folder_path):
    # Empty image list
    image_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp')):  # Check for image file extensions
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            # Read the image using OpenCV
            image_list.append(file_path)
    return image_list


def canny_detect(img_path):
    img = cv2.imread(img_path)
    if img is not None:
        # To grayscale
        gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # To binary image - Original
        orig_binary_image = np.where(gray_original > 0, 1, 0)
        # Sum of ones - Original
        orig_sum_of_ones = np.sum(orig_binary_image)

        # Canny edge detect
        canny_edges = cv2.Canny(gray_original, 100, 256)
        # To binary image - Canny
        canny_binary_image = np.where(canny_edges > 0, 1, 0)
        # Sum of ones - Canny
        canny_sum_of_ones = np.sum(canny_binary_image)
        return img, gray_original, canny_edges, canny_binary_image, orig_binary_image, canny_sum_of_ones, orig_sum_of_ones
    else:
        return None


# Zpracování obrázků a zobrazení v jednom okně
def visualization(img_list):
    fig, axs = plt.subplots(2, len(img_list), figsize=(16, 6))

    for i, image_file in enumerate(img_list):
        results = canny_detect(image_file)

        if results is not None:
            # Extract values from result
            img, gray_original, canny_edges, canny_binary_image, orig_binary_image, canny_sum_of_ones, \
            orig_sum_of_ones = results
            # original, gray, canny, binary, sum_of_ones = results

            axs[0, i].imshow(img)
            axs[0, i].set_title(f'Original\nSum: {orig_sum_of_ones}')
            axs[0, i].axis('off')

            axs[1, i].imshow(canny_binary_image, cmap='gray')
            axs[1, i].set_title(f'Canny Edges\nSum: {canny_sum_of_ones}')
            axs[1, i].axis('off')
        else:
            axs[0, i].axis('off')
            axs[1, i].axis('off')

    plt.show()


img_list = create_image_list('PVI_C04')
visualization(img_list)
print('done')
