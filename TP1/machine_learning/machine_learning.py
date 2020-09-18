import cv2
import numpy as np

from TP1.machine_learning.utils.dataset import int_to_label
from TP1.open_cv.tp1 import get_greatest_contour


def on_trackbar_change(val):
    pass


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    saved_shape = []
    window_name = 'Machine Learning - Shape detector'
    model_path = './utils/svm_data.dat'
    trackbar_name = 'Trackbar'

    classifier = cv2.ml.SVM_load(model_path)
    cv2.namedWindow(window_name)
    trackbar = cv2.createTrackbar(trackbar_name, window_name, 127, 255, on_trackbar_change)
    while True:
        _, frame = cap.read()
        flipped = cv2.flip(frame, 1)
        gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
        trackbar_val = cv2.getTrackbarPos(trackbar_name, window_name)
        _, thresh = cv2.threshold(gray, trackbar_val, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(flipped, contours, -1, (0, 0, 255), 6)
        new_frame = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)

        if len(contours) > 0:
            greatest_contour = get_greatest_contour(contours)

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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
