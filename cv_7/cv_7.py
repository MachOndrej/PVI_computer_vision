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


def compare_vectors(sample_vector, alphabet_vector_dir):
    # Abeceda (prvni misto blank space)
    alphabet_uppercase = [chr(i) for i in range(65, 91)]
    alphabet_uppercase.insert(0, ' ')
    smallest_euclidean_distance = 100000000000000000000000000000
    for i in range(0, len(alphabet_vector_dir)):
        euclidean_distance = distance.euclidean(sample_vector, alphabet_vector_dir[i])
        if smallest_euclidean_distance > euclidean_distance:
            smallest_euclidean_distance = euclidean_distance
            alphabet_letter_position = i
            # FIX X, Y, Z
            if alphabet_letter_position == 24:
                alphabet_letter_position += 1
            elif alphabet_letter_position == 25:
                alphabet_letter_position += 1

    letter = alphabet_uppercase[alphabet_letter_position]
    return letter


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

    alphabet_vector_dir = []
    for i in range(0, len(bin_img_dir)):
        # Výpočet horizontální projekce
        h_proj = np.sum(bin_img_dir[i], axis=1)
        # Výpočet vertikální projekce
        v_proj = np.sum(bin_img_dir[i], axis=0)
        sample_vector = np.append(v_proj, h_proj)
        alphabet_vector_dir.append(sample_vector)
    # Fix last "_" position
    last_val = alphabet_vector_dir.pop()
    alphabet_vector_dir.insert(0, last_val)
    # Zpracovani obrazku s textem
    img = cv2.imread('pvi_cv07_text.bmp')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # to gray
    _, bin_img_rev = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)  # Apply threshold
    bin_img = np.zeros_like(bin_img_rev)        # prohozeni 0 a 1
    bin_img[bin_img_rev == 0] = 1
    # Labeling
    s = np.ones((3, 3))
    BWlabel, ncc = label(bin_img, structure=s)

    trimmed_label = BWlabel[2:-2, 2:-2]     # zbaveni se okraje [rows, columns]
    text = ""                               # string pro vypsani
    start = 0                               # pocatecni souradnice
    for letter in range(1, ncc+1):
        part = trimmed_label[:, start:start+5]
        sample_vector = []
        part[part == letter] = 1
        # Výpočet horizontální projekce
        h_proj = np.sum(part, axis=1)
        # Výpočet vertikální projekce
        v_proj = np.sum(part, axis=0)
        sample_vector = np.append(v_proj, h_proj)     # vector pro porovnani s alphabet_vector_dir
        # Najdu nejmensi vzdalenost
        corresponding_letter = compare_vectors(sample_vector, alphabet_vector_dir)
        # pridam prislusne pismeno
        text += corresponding_letter
        # posunu se po textu
        start += 7
    print('\n', text)


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    intersection_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    union_area = w1 * h1 + w2 * h2 - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def face_detect_part():
    filImage = 'pvi_cv07_people.jpg'
    bgr = cv2.imread(filImage)
    gray = cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY)

    boxes = []
    with open('pvi_cv07_boxes_01.txt') as f:
        lines = f.read().splitlines()
        for line in lines:
            vec = line.split(' ')
            vec = [int(x) for x in vec]
            boxes.append(vec)

    for (x, y, w, h) in boxes:
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
    faceCascade = cv2.CascadeClassifier('pvi_cv07_haarcascade_frontalface_default.xml')  # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.4,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # dorovnani kvuli nerozpoznanym oblicejum
    list_diff = abs(len(faces) - len(boxes))
    for _ in range(list_diff):
        row_to_add = np.array([0, 0, 0, 0])
        faces = np.vstack([faces, row_to_add])

    true_positives = 0
    false_positives = 0
    computed_IoUs_list = []

    for (x_detected, y_detected, w_detected, h_detected) in faces:
        detected_box = [x_detected, y_detected, w_detected, h_detected]
        detected_iou = max(calculate_iou(detected_box, true_box) for true_box in boxes)
        computed_IoUs_list.append(detected_iou)
    # vyhodnocení pro IoU = 0,5 - accuracy, precision a recall
    IoU = 0.5
    for jj in computed_IoUs_list:
        if jj <= IoU:
            false_positives += 1
        else:
            true_positives += 1

    false_negatives = len(boxes) - true_positives - false_positives
    total_detected = true_positives + false_positives

    # acc = (TP + TN)/(TP + TN + FP + FN) = 5/10 = 0.5
    accuracy = true_positives / total_detected
    # prec = TP/(TP + FP) = 5/10 = 0.5
    precision = true_positives / (true_positives + false_positives)
    # rec = TP/(TP + FN) = 5/
    recall = true_positives / (true_positives + false_negatives)
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.show()


if __name__ == "__main__":
    main()
    face_detect_part()
