import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import fft
from scipy import ndimage


def bounding_box(points):
    min_x = min(points[1])
    max_x = max(points[1])
    min_y = min(points[0])
    max_y = max(points[0])
    return (min_x, min_y), (max_x, max_y)


def main():
    reference = cv2.imread('pvi_cv10_vzor_pomeranc.bmp')
    reference_hue = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)[:, :, 0]
    reference_hist = cv2.calcHist([reference_hue], [0], None, [256], [0, 256])
    reference_hist = reference_hist / np.max(reference_hist)
    reference_shape = reference_hue.shape
    ref_half_y = int(reference_shape[0] / 2)
    ref_half_x = int(reference_shape[1] / 2)

    threshold = 10000000
    cap = cv2.VideoCapture('pvi_cv10_video_in.mp4')
    ret, frame = cap.read()
    if ret is None or frame is None:
        print('End Of File')
        return
    ROI_bbox = ((0, 0), (frame.shape[0], frame.shape[1]))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('pvi_cv10_video_out.avi', fourcc, 10.0, (frame.shape[1], frame.shape[0]))

    background = frame.copy()
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    prev_frame = frame.copy()
    frame_number = 0
    scene_number = 0
    while cap.isOpened():

        ret, frame = cap.read()
        if ret is None or frame is None:
            print('End Of File')
            return

        diff = cv2.absdiff(prev_frame, frame)
        diff_sum = cv2.sumElems(diff)[0]  # Calculate the sum of absolute differences

        if diff_sum > threshold:
            print(f"Scene change detected at frame {frame_number} with diff {diff_sum}")
            scene_number += 1
        prev_frame = frame.copy()
        frame_number += 1

        if scene_number in [0, 2]:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff_background = background_gray.astype(np.int32) - frame_gray.astype(np.int32)
            diff_background[diff_background < 30] = 0
            diff_background[diff_background >= 30] = 1
            diff_background = cv2.erode(diff_background.astype(np.uint8), np.ones((3, 3)), iterations=1)
            points = np.where(diff_background == 1)

            if len(points[0]) > 0:
                bbox = bounding_box(points)
                frame = cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 1, 1)
                out.write(frame)

        if scene_number in [1]:
            frame_hue = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 0]
            ROI = frame_hue[ROI_bbox[0][0]:ROI_bbox[1][0], ROI_bbox[0][1]:ROI_bbox[1][1]]
            p = reference_hist[ROI]
            p = p[:, :, 0]
            center_of_mass = ndimage.center_of_mass(p)
            x = int(center_of_mass[0]) + ROI_bbox[0][0]
            y = int(center_of_mass[1]) + ROI_bbox[0][1]
            bbox = ((int(y - ref_half_y), int(x - ref_half_x)),
                    (int(y + ref_half_y), int(x + ref_half_x)))
            frame = cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 1, 1)
            ROI_bbox = bbox
            out.write(frame)

        cv2.imshow('Video with Bounding Boxes', frame)

        # Break the loop if the user presses 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release the video capture object outside the loop
    cap.release()
    out.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
