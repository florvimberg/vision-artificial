import cv2

from tp1ShapeDetector.machine_learning.utils.dataset import int_to_label
from tp1ShapeDetector.tp1 import get_greatest_contour
from tp1ShapeDetector.utils.common_utils import print_hu_moments

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    saved_shape = []
    window_name = 'Machine Learning - Shape detector'
    model_path = './utils/svm_data.dat'

    classifier = cv2.ml.SVM_load(model_path)

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        trackbar_val = cv2.getTrackbarPos('Trackbar', window_name)
        _, thresh = cv2.threshold(gray, trackbar_val, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        new_frame = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)

        if len(contours) > 0:
            greatest_contour = get_greatest_contour(contours)

            moments = cv2.moments(greatest_contour)
            hu_moments = cv2.HuMoments(moments)

            prediction = classifier.predict(hu_moments)[1]
            print(int_to_label(prediction))

        cv2.imshow(window_name, cv2.flip(frame, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
