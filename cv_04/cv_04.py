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
    # Převod do grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 2D Fourierova transformace
    dft = np.fft.fft2(image)
    # Posunout nulovou frekvenci do středu
    dft_shifted = np.fft.fftshift(dft)
    # Amplitudové spektrum (logaritmus z absolutní hodnoty)
    amplitude_spectrum = np.log(np.abs(dft_shifted) + 1)  # Přidáme 1, aby se vyhnuli log(0)
    # Zobrazíme amplitudové spektrum s colorbarem
    plt.imshow(amplitude_spectrum, cmap='jet')
    plt.colorbar()
    plt.title('Spectrum')
    plt.show()


#plot_amplitude_spectrum(img)

# 2) Porovná obrázky na základě nejmenší vzdálenosti
# příznakových vektorů, kde příznakovým vektorem bude:
# a. histogram počítaný z šedotónového obrázku
# TODO histogram grayscale obrayku, porovnat vzdalenost vektoru
im = cv2.imread('PVI_CV03/pvi_cv03_im01.jpg')
im2 = cv2.imread('PVI_CV03/pvi_cv03_im02.jpg')
im3 = cv2.imread('PVI_CV03/pvi_cv03_im03.jpg')

hist1 = cv2.calcHist([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])

dctM = dctn(imG)
R = 5
dctRvec = dctM[0:R, 0:R].flatten()
distGH[j] = np.linalg.norm(vecKGH - vecUGH)
imIDsGH = np.argsort(distGH)

# Calculate a similarity or distance metric, for example, Bhattacharyya distance
distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
distance2 = cv2.compareHist(hist1, hist3, cv2.HISTCMP_BHATTACHARYYA)
#TODO porovnavat  distGH[j] = np.linalg.norm(vecKGH - vecUGH)

# Output the distance (lower values indicate greater similarity)
print("Bhattacharyya Distance:", distance, distance2)

# b. histogram počítaný z Hue barevné složky obrázku
# c. vektor z oblasti 5x5 z 2D DCT, kolem b
