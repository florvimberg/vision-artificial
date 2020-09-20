import cv2
import numpy as np

from common.common_utils import filter_contours
from machine_learning.utils.dataset import int_to_label
from open_cv.tp1 import get_greatest_contour


def on_trackbar_change(val):
    pass


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    saved_shape = []
    window_name = 'Machine Learning - Shape detector'
    model_path = './utils/svm_data.dat'
    trackbar_name = 'Threshold Trackbar'
    cv2.namedWindow(window_name)

    # Max area trackbar
    trackbar_name3 = 'Max Area Trackbar'
    slider_max3 = 100000
    cv2.createTrackbar(trackbar_name3, window_name, 90000, slider_max3, on_trackbar_change)

    # Min area trackbar
    trackbar_name4 = 'Min Area Trackbar'
    slider_max4 = 10000
    cv2.createTrackbar(trackbar_name4, window_name, 1, slider_max4, on_trackbar_change)

    classifier = cv2.ml.SVM_load(model_path)
    trackbar = cv2.createTrackbar(trackbar_name, window_name, 127, 255, on_trackbar_change)
    while True:
        _, frame = cap.read()
        flipped = cv2.flip(frame, 1)
        gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
        trackbar_val = cv2.getTrackbarPos(trackbar_name, window_name)
        _, thresh = cv2.threshold(gray, trackbar_val, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        max_area = cv2.getTrackbarPos(trackbar_name3, window_name)
        min_area = cv2.getTrackbarPos(trackbar_name4, window_name)
        filtered_contours = filter_contours(contours, min_area, max_area)
        cv2.drawContours(flipped, filtered_contours, -1, (0, 0, 255), 6)

        if len(filtered_contours) > 0:
            greatest_contour = get_greatest_contour(filtered_contours)

            cv2.drawContours(flipped, greatest_contour, -1, (0, 255, 0), 12)

            moments = cv2.moments(greatest_contour)
            hu_moments = cv2.HuMoments(moments)

            # Recibe un [][] donde el segundo array es de tamaño 7
            sample = np.array([hu_moments.ravel()], dtype=np.float32)
            prediction = classifier.predict(sample)

            label = int_to_label(prediction[1][0])
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            cv2.putText(flipped, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow(window_name, flipped)
        cv2.imshow('debug', closing)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
