import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.ndimage import label
from scipy.spatial import distance


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


def main():
    # Ziskani vectoru pismen
    img_dir = create_image_list('dir_znaky')
    bin_img_dir = []
    for obrazek_file in img_dir:
        obrazek = cv2.imread(obrazek_file)
        gray = cv2.cvtColor(obrazek, cv2.COLOR_BGR2GRAY)                 # to gray
        _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)       # Apply threshold
        reverse = np.zeros_like(binary)
        reverse[binary == 0] = 1
        bin_img_dir.append(reverse)

    vector_dir = []
    for i in range(-1, len(bin_img_dir)):
        # Výpočet horizontální projekce
        h_proj = np.sum(bin_img_dir[i], axis=1)
        # Výpočet vertikální projekce
        v_proj = np.sum(bin_img_dir[i], axis=0)

        vectors = np.append(v_proj, h_proj)
        vector_dir.append(vectors)
        #print(vectors)

    # Zpracovani obrazku s textem
    img = cv2.imread('pvi_cv07_text.bmp')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # to gray
    _, bin_img_rev = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)  # Apply threshold
    bin_img = np.zeros_like(bin_img_rev)        # prohozeni 0 a 1
    bin_img[bin_img_rev == 0] = 1

    s = np.ones((3,3))
    BWlabel, ncc = label(bin_img, structure=s)

    trimmed_label = BWlabel[2:-2, 2:-2]     # zbaveni se okraje [rows, columns]

    start = 0
    #for letter in range(1, ncc):
    for letter in range(1, 3):
        part = trimmed_label[:, start:start+5]
        vectors = []
        part[part == letter] = 1
        print(part)

        # Výpočet horizontální projekce
        h_proj = np.sum(part, axis=1)
        # Výpočet vertikální projekce
        v_proj = np.sum(part, axis=0)
        vectors = np.append(v_proj, h_proj)
        print(vectors)
        start += 7

    # TODO: vydalenost vektoru
    # Define two points in 3D space
    point1 = (1, 2, 3)
    point2 = (4, 5, 6)

    # Calculate the Euclidean distance between the two points
    euclidean_distance = distance.euclidean(point1, point2)


if __name__ == "__main__":
    main()
