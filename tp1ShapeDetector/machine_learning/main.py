import cv2

from tp1ShapeDetector.tp1 import get_greatest_contour
from tp1ShapeDetector.utils.common_utils import print_hu_moments

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    saved_shape = []
    window_name = 'Machine Learning - Shape detector'

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

            sensibility_trackbar_val = cv2.getTrackbarPos('Sensibility trackbar', window_name)
            cv2.drawContours(new_frame, [greatest_contour], -1, (255, 0, 0), 12)

            if cv2.waitKey(1) & 0xFF == ord('f'):
                saved_shape = greatest_contour

            if cv2.waitKey(1) & 0xFF == ord('p'):
                print_hu_moments(greatest_contour, saved_shape)

            # 6. Clasificación binaria: Aplicar un criterio sobre los invariantes para decidir si el contorno corresponde o no a la pieza a reconocer
            for contour in contours:
                # compares hu moments from two contours
                # 7. Anotar los contornos sobre la imagen monocromática y visualizarla
                if len(saved_shape) > 0 and cv2.matchShapes(contour, saved_shape, cv2.CONTOURS_MATCH_I2, 0) < (
                        sensibility_trackbar_val / 100):
                    # Todos los contornos cuyos momentos de Hu sean considerados similares a el contorno guardado se muestran en color verde.
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
                else:
                    # Todos los demas contornos se muestran con un contorno en rojo.
                    cv2.drawContours(frame, [contour], -1, (0, 0, 255), 3)

        cv2.imshow(window_name, cv2.flip(frame, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
