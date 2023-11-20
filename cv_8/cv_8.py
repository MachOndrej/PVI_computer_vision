import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
from scipy.ndimage import label
from scipy.ndimage import rotate
import easyocr
from scipy.stats import entropy


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    w1 -= x1
    h1 -= y1
    x2, y2, w2, h2 = box2
    w2 -= x2
    h2 -= y2

    intersection_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    union_area = w1 * h1 + w2 * h2 - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def spz_detection_task():
    spz_image = cv2.imread('pvi_cv08/pvi_cv08_spz.png')
    spz_image_rgb = cv2.cvtColor(spz_image, cv2.COLOR_BGR2RGB)
    spz_image_gray = cv2.cvtColor(spz_image, cv2.COLOR_BGR2GRAY)
    spz_image_segmented = cv2.threshold(spz_image_gray, 250, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((5, 5))
    spz_image_no_text = spz_image_segmented.copy()
    spz_image_no_text = cv2.morphologyEx(spz_image_no_text, cv2.MORPH_OPEN, kernel, iterations=1)
    spz_image_no_text = cv2.morphologyEx(spz_image_no_text, cv2.MORPH_CLOSE, kernel, iterations=5)
    spz_image_no_text = np.invert(spz_image_no_text)
    spz_image_no_text[spz_image_no_text < 255] = 0

    dst = cv2.cornerHarris(spz_image_no_text, 5, 7, 0.04)
    dst = cv2.dilate(dst, None)
    img = spz_image_rgb.copy()
    img[dst > 0.01 * dst.max()] = [255, 0, 0]

    # finding center coordinates
    dst[dst <= 0.01 * dst.max()] = 0.
    dst[dst > 0] = 1.
    labels_dst, ncc = label(dst)
    nonzero_labels_coords = np.nonzero(dst)
    nonzero_labels = labels_dst[nonzero_labels_coords]
    centers = []
    for i in range(1, ncc + 1):
        yy = nonzero_labels_coords[0][nonzero_labels == i]
        xx = nonzero_labels_coords[1][nonzero_labels == i]
        yy_center = np.mean(yy, dtype=np.int32)
        xx_center = np.mean(xx, dtype=np.int32)
        centers.append([xx_center, yy_center])
    centers = np.array(centers)

    rotation_radians = np.arctan((centers[0, 1] - centers[2, 1]) / (centers[0, 0] - centers[2, 0]))
    rotation_degrees = np.rad2deg(rotation_radians)

    spz_image_rgb_rotated = rotate(spz_image_rgb, rotation_degrees)

    reader = easyocr.Reader(['cs'], gpu=False)
    text = reader.readtext(spz_image_rgb_rotated)
    full_text = ''
    for i in range(len(text)):
        full_text += text[i][1] + ' '
    full_text = full_text[0:-1]

    cv2.putText(spz_image_rgb_rotated, full_text, [20, 100], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    plt.subplot(2, 2, 1)
    plt.imshow(spz_image_rgb)
    plt.title('Orig. Im.')

    plt.subplot(2, 2, 2)
    plt.imshow(spz_image_segmented, cmap='jet')
    plt.title('Bin. Im.')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.imshow(spz_image_no_text, cmap='jet')
    plt.title('Bin. Im. - Result')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.imshow(img)
    plt.title('Im. + Harris')

    plt.show()

    plt.imshow(spz_image_rgb_rotated)
    plt.show()


def sunflower_detector_task(sunflower_path, b_boxes_path, entropy_thresh=1.8,
                            min_sigma=10, max_sigma=100, num_sigma=20, threshold=0.15, overlap=0.5):
    # template read
    template = cv2.imread('pvi_cv08/pvi_cv08_sunflower_template.jpg')
    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    template_hue = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)[:, :, 0]
    hist_template = cv2.calcHist([template_hue], [0], None, [256], [0, 256])
    hist_template += 0.0001

    # image read and color conversions
    sunflower = cv2.imread(sunflower_path)
    sunflower_gray = cv2.cvtColor(sunflower, cv2.COLOR_BGR2GRAY)
    sunflower_gray = np.invert(sunflower_gray)
    sunflower_rgb_blobs = cv2.cvtColor(sunflower, cv2.COLOR_BGR2RGB)
    sunflower_rgb_detected_flowers = sunflower_rgb_blobs.copy()
    sunflower_rgb_annotated_flowers = sunflower_rgb_blobs.copy()

    # loading annotated bounding boxes
    ground_truth_rectangles = []
    with open(b_boxes_path, 'r') as file:
        while line := file.readline():
            line_split = line.split(' ')
            rectangle = [int(line_split[0]), int(line_split[1]), int(line_split[2]), int(line_split[3])]
            ground_truth_rectangles.append(
                rectangle
            )
            # draw annotated boxes
            cv2.rectangle(sunflower_rgb_annotated_flowers, [rectangle[0], rectangle[1]], [rectangle[2], rectangle[3]],
                          [255, 255, 255], 2)

    # finding blobs
    blobs = skimage.feature.blob_log(sunflower_gray, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma,
                                     threshold=threshold, overlap=overlap, log_scale=False, threshold_rel=None,
                                     exclude_border=False)

    # initialization
    blob_cutouts = []
    blobs_accepted = []
    rectangles = []
    rectangles_accepted = []
    computed_IoUs_list = []
    for idx, blob in enumerate(blobs):
        # unpacking
        center_y, center_x, sigma = blob

        # rectangle cutout
        radius = int(sigma * 2)
        rectangle = [max(int(center_x - radius), 0), max(int(center_y - radius), 0),
                     min(int(center_x + radius), sunflower.shape[1]), min(int(center_y + radius), sunflower.shape[0])]
        cutout = sunflower[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2], :]
        blob_cutouts.append(cutout)
        rectangles.append(rectangle)

        # blob circle draw
        cv2.circle(sunflower_rgb_blobs, (int(center_x), int(center_y)), radius, [255, 0, 255], 2)

        # cvt to hue and histogram
        cutout_hue = cv2.cvtColor(cutout, cv2.COLOR_BGR2HSV)[:, :, 0]
        hist = cv2.calcHist([cutout_hue], [0], None, [256], [0, 256])
        hist += 0.0001

        # calculating entropy
        calculated_entropy = entropy(hist, hist_template)
        if calculated_entropy < entropy_thresh:
            blobs_accepted.append(idx)
            rectangles_accepted.append(rectangle)
            # draw detected rectangles
            cv2.rectangle(sunflower_rgb_detected_flowers, [rectangle[0], rectangle[1]], [rectangle[2], rectangle[3]],
                          [255, 0, 255], 2)
            # IOU calc
            detected_iou = max(calculate_iou(rectangle, true_rect) for true_rect in ground_truth_rectangles)
            computed_IoUs_list.append(detected_iou)

    # IoU
    true_positives = 0
    false_positives = 0
    IoU = 0.5
    for jj in computed_IoUs_list:
        if jj <= IoU:
            false_positives += 1
        else:
            true_positives += 1

    false_negatives = len(ground_truth_rectangles) - true_positives - false_positives

    # prec = TP/(TP + FP) = 5/10 = 0.5
    precision = true_positives / (true_positives + false_positives)
    # rec = TP/(TP + FN) = 5/5 = 1
    recall = true_positives / (true_positives + false_negatives)

    print(f"Precision: {true_positives}/({true_positives}+{false_positives}) = {precision}")
    print(f"Recall: {true_positives}/({true_positives}+{false_negatives}) = {recall}")

    # show template
    plt.subplot(1, 2, 1)
    plt.imshow(template_rgb)
    plt.subplot(1, 2, 2)
    plt.plot(hist_template)
    plt.grid()
    plt.show()

    # show blobs
    plt.imshow(sunflower_rgb_blobs)
    plt.show()

    # show detected and annotated
    plt.subplot(1, 2, 1)
    plt.imshow(sunflower_rgb_detected_flowers)
    plt.title("Detected Flowers")
    plt.subplot(1, 2, 2)
    plt.imshow(sunflower_rgb_annotated_flowers)
    plt.title("Annotated Flowers")
    plt.show()


def main():
    spz_detection_task()

    min_sigma = 10
    max_sigma = 100
    num_sigma = 20
    threshold = 0.15
    overlap = 0.5

    entropy_thresh = 1.5

    sunflower_path = 'pvi_cv08/pvi_cv08_sunflowers1.jpg'
    b_boxes_path = 'pvi_cv08/pvi_cv08_sunflowers1.txt'
    print(sunflower_path)
    sunflower_detector_task(sunflower_path, b_boxes_path, entropy_thresh=entropy_thresh, min_sigma=min_sigma,
                            max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold, overlap=overlap)
    print()

    sunflower_path = 'pvi_cv08/pvi_cv08_sunflowers2.jpg'
    b_boxes_path = 'pvi_cv08/pvi_cv08_sunflowers2.txt'
    print(sunflower_path)
    sunflower_detector_task(sunflower_path, b_boxes_path, entropy_thresh=entropy_thresh, min_sigma=min_sigma,
                            max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold, overlap=overlap)
    print()

    sunflower_path = 'pvi_cv08/pvi_cv08_sunflowers3.jpg'
    b_boxes_path = 'pvi_cv08/pvi_cv08_sunflowers3.txt'
    print(sunflower_path)
    sunflower_detector_task(sunflower_path, b_boxes_path, entropy_thresh=entropy_thresh, min_sigma=min_sigma,
                            max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold, overlap=overlap)
    print()

    sunflower_path = 'pvi_cv08/pvi_cv08_sunflowers4.jpg'
    b_boxes_path = 'pvi_cv08/pvi_cv08_sunflowers4.txt'
    print(sunflower_path)
    sunflower_detector_task(sunflower_path, b_boxes_path, entropy_thresh=entropy_thresh, min_sigma=min_sigma,
                            max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold, overlap=overlap)
    print()


if __name__ == "__main__":
    main()
