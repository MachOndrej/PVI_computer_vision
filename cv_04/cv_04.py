import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_amplitude_spectrum(image):
    # 2D Fourierova transformace
    dft = np.fft.fft2(image)
    # Posunout nulovou frekvenci do středu
    dft_shifted = np.fft.fftshift(dft)
    # Amplitudové spektrum (logaritmus z absolutní hodnoty)
    amplitude_spectrum = np.log(np.abs(dft_shifted) + 1)  # Přidáme 1, aby se vyhnuli log(0)
    return amplitude_spectrum


def median_filter(data, filter_size):   # data=image, filtersize=e.g.:5
    temp = []       # to store values
    indexer = filter_size // 2  # half of filter -> to set offset from center of window filter
    data_final = np.zeros((len(data), len(data[0])))    # to store filtered image
    for i in range(len(data)):
        for j in range(len(data[0])):
            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)      # out of range -> append 0
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)      # out of range -> append 0
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer]) # append to temp
            temp.sort()     # sort temp
            data_final[i][j] = temp[len(temp) // 2]  # find median in tem and add to final
            temp = []
    data_final = data_final.astype(np.uint8)    # conversion to uint8
    return data_final


def visualization(image_list, first_img_name, second_img_name, title):
    # Start suobplot
    fig, axs = plt.subplots(2, 3, constrained_layout=True)
    # Set up for visualiyation - 1st row    (axs[row,col])
    axs[0, 0].imshow(image_list[0], cmap='gray')
    axs[0, 0].set_title(first_img_name, fontsize=10.0)
    axs[0, 1].plot(image_list[1])
    axs[0, 1].set_title('Histogram', fontsize=10.0)
    axs[0, 1].set_xlabel("gray-scale value", fontsize=8.0)
    axs[0, 2].set_title('Spectrum', fontsize=10.0)
    # 2nd row
    axs[1, 0].imshow(image_list[3], cmap='gray')
    axs[1, 0].set_title(second_img_name, fontsize=10.0)
    axs[1, 1].plot(image_list[4])
    axs[1, 1].set_title('Histogram', fontsize=10.0)
    axs[1, 1].set_xlabel("gray-scale value", fontsize=8.0)
    axs[1, 2].set_title('Spectrum', fontsize=10.0)
    # Set up colorbars
    fig.colorbar(axs[0, 2].imshow(image_list[2], cmap='jet'), ax=axs[0, 2], orientation='vertical')
    fig.colorbar(axs[1, 2].imshow(image_list[5], cmap='jet'), ax=axs[1, 2], orientation='vertical')
    # Set up title
    fig.suptitle(title, fontsize=10.0)
    plt.show()


def main():
    img = cv2.imread('pvi_cv04.png')
    # Preparation of original image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    gray_amp_spec = plot_amplitude_spectrum(gray)

    # 1a) Averaging
    aver_img = cv2.blur(gray, (3, 3))
    aver_img_hist = cv2.calcHist([aver_img], [0], None, [256], [0, 256])
    aver_img_amp_spec = plot_amplitude_spectrum(aver_img)
    image_list1 = [gray, gray_hist, gray_amp_spec, aver_img, aver_img_hist, aver_img_amp_spec]
    visualization(image_list1, 'Original Image', 'New Image', 'Image smoothing - Averaging')

    # 1b) Median - OpenCV
    med_img = cv2.medianBlur(gray, 5)
    med_img_hist = cv2.calcHist([med_img], [0], None, [256], [0, 256])
    med_img_amp_spec = plot_amplitude_spectrum(med_img)
    image_list2 = [gray, gray_hist, gray_amp_spec, med_img, med_img_hist, med_img_amp_spec]
    visualization(image_list2, 'Original Image', 'New Image', 'Image smoothing - Median - OpenCV')

    # 1c) Median x My median
    my_med = median_filter(gray, 5)
    my_med_hist = cv2.calcHist([my_med], [0], None, [256], [0, 256])
    my_med_amp_spec = plot_amplitude_spectrum(my_med)
    image_list3 = [med_img, med_img_hist, med_img_amp_spec, my_med, my_med_hist, my_med_amp_spec]
    visualization(image_list3, 'OpenCV Median', 'My Median', 'Image smoothing - Median - My')


if __name__ == "__main__":
    main()
