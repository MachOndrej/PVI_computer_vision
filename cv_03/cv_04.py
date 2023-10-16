import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.fft import dctn


def create_image_list(folder_path):
    # Empty image list
    image_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp')):  # Check for image file extensions
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            # Read the image using OpenCV
            image = cv2.imread(file_path)
            if image is not None:
                # Append the image to the list
                image_list.append(image)
    return image_list


# 1) Vypočte a zobrazí 2D amplitudové spektrum
# (z 2D DFT) pro libovolný (šedotónový) obrázek
def plot_amplitude_spectrum(image):
    # 2D Fourierova transformace
    dft = np.fft.fft2(image)
    # Posunout nulovou frekvenci do středu
    dft_shifted = np.fft.fftshift(dft)
    # Amplitudové spektrum (logaritmus z absolutní hodnoty)
    amplitude_spectrum = np.log(np.abs(dft_shifted) + 1)  # Přidáme 1, aby se vyhnuli log(0)
    # Zobrazíme amplitudové spektrum s colorbarem
    return amplitude_spectrum


#plot_amplitude_spectrum(img)

# 2) Porovná obrázky na základě nejmenší vzdálenosti
# příznakových vektorů, kde příznakovým vektorem bude:
def main():
    bgr = cv2.imread('PVI_CV03/pvi_cv03_im09.jpg')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    amplitude_spectrum = plot_amplitude_spectrum(gray)
    plt.imshow(rgb)
    plt.show()
    plt.imshow(amplitude_spectrum, cmap='jet')
    plt.colorbar()
    plt.title('Spectrum')
    plt.show()

    list_image = create_image_list("PVI_CV03")
    list_image_rgb = []
    list_hist_gray = []
    list_hist_hue = []
    list_dct = []
    for image in list_image:
        list_image_rgb.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        gray_temp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        list_hist_gray.append(cv2.calcHist([gray_temp], [0], None, [256], [0, 256]))

        hue_temp = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        list_hist_hue.append(cv2.calcHist([hue_temp], [0], None, [256], [0, 256]))

        dct_temp = dctn(gray_temp)
        list_dct.append(dct_temp[0:5, 0:5].flatten())

    image_list_len = len(list_image)
    results_hist_gray = []
    results_hist_hue = []
    results_dct = []
    for ii in range(image_list_len):
        dist_gray = np.zeros((image_list_len - 1, 1))
        temp_list_hist_gray = list_hist_gray[:ii] + list_hist_gray[ii+1:]

        dist_hue = np.zeros((image_list_len - 1, 1))
        temp_list_hist_hue = list_hist_hue[:ii] + list_hist_hue[ii+1:]

        dist_dct = np.zeros((image_list_len - 1, 1))
        temp_list_dct = list_dct[:ii] + list_dct[ii+1:]

        for jj in range(image_list_len - 1):
            dist_gray[jj] = np.linalg.norm(list_hist_gray[ii] - temp_list_hist_gray[jj])
            dist_hue[jj] = np.linalg.norm(list_hist_hue[ii] - temp_list_hist_hue[jj])
            dist_dct[jj] = np.linalg.norm(list_dct[ii] - temp_list_dct[jj])
        results_hist_gray.append(dist_gray.argsort(axis=0))
        results_hist_hue.append(dist_hue.argsort(axis=0))
        results_dct.append(dist_dct.argsort(axis=0))

    rows = columns = image_list_len

    fig_gray, axes_gray = plt.subplots(nrows=rows, ncols=columns, figsize=(columns * 3, rows * 3), constrained_layout=True)
    fig_gray.suptitle("gray", fontsize=60)
    for row in range(1, rows + 1):
        fig_gray.add_subplot(rows, columns, row + (8 * (row - 1)))
        plt.imshow(list_image_rgb[row - 1])
        temp_image_list = list_image_rgb[:row-1] + list_image_rgb[row:]
        for column in range(2, columns + 1):
            img_idx = results_hist_gray[row - 1][column-2][0]
            fig_gray.add_subplot(rows, columns, row + (8 * (row - 1)) + column - 1)
            plt.imshow(temp_image_list[img_idx])
    fig_gray.show()

    fig_hue, axes_hue = plt.subplots(nrows=rows, ncols=columns, figsize=(columns * 3, rows * 3), constrained_layout=True)
    fig_hue.suptitle("hue", fontsize=60)
    for row in range(1, rows + 1):
        fig_hue.add_subplot(rows, columns, row + (8 * (row - 1)))
        plt.imshow(list_image_rgb[row - 1])
        temp_image_list = list_image_rgb[:row-1] + list_image_rgb[row:]
        for column in range(2, columns + 1):
            img_idx = results_hist_hue[row - 1][column-2][0]
            fig_hue.add_subplot(rows, columns, row + (8 * (row - 1)) + column - 1)
            plt.imshow(temp_image_list[img_idx])
    fig_hue.show()

    fig_dct, axes_dct = plt.subplots(nrows=rows, ncols=columns, figsize=(columns * 3, rows * 3), constrained_layout=True)
    fig_dct.suptitle("dct", fontsize=60)
    for row in range(1, rows + 1):
        fig_dct.add_subplot(rows, columns, row + (8 * (row - 1)))
        plt.imshow(list_image_rgb[row - 1])
        temp_image_list = list_image_rgb[:row-1] + list_image_rgb[row:]
        for column in range(2, columns + 1):
            img_idx = results_dct[row - 1][column-2][0]
            fig_dct.add_subplot(rows, columns, row + (8 * (row - 1)) + column - 1)
            plt.imshow(temp_image_list[img_idx])
    fig_dct.show()


if __name__ == "__main__":
    main()
