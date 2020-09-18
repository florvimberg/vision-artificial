# 1. Convertir la imagen a monocromática
# 2. Aplicar un threshold con umbral ajustable con una barra de desplazamiento
#   se pueden incluir opciones de ajuste automático
# 3. Aplicar operaciones morfológicas para eliminar ruido de la imagen
# 4. Si se obtienen varios contornos, elegir con algún criterio al menos uno para procesar
#   es preferible procesar todos los contornos significativos, evitando contornos provenientes de ruido
# 5. Obtener parámetros del contorno
#   invariantes de Hu
#   otros momentos, como m00
#   otros parámetros relevantes, como concavidades
# 6. Clasificación binaria: Aplicar un criterio sobre los invariantes para decidir si el contorno corresponde o no a la pieza a reconocer
# 7. Anotar los contornos sobre la imagen monocromática y visualizarla
#   usar verde para el contorno reconocido
#   usar rojo para los contornos desconocidos
#   la forma de anotación queda a criterio de los alumnos
import cv2

if __name__ == '__main__': print('TP 1')


def on_trackbar_change(val):
    pass


window_name = 'Shape detector'
trackbar_name = 'Trackbar'
cv2.namedWindow(window_name)
slider_max = 150
cv2.createTrackbar(trackbar_name, window_name, 50, slider_max, on_trackbar_change)

trackbar_name2 = 'Sensibility trackbar'
cv2.namedWindow(window_name)
slider_max2 = 100
cv2.createTrackbar(trackbar_name2, window_name, 10, slider_max2, on_trackbar_change)

# Da como resultado el contorno con el area maxima dada una lista de contornos.
def get_greatest_contour(contours):
    max_contour = contours[0]
    for contour in contours:
        if cv2.contourArea(contour) > cv2.contourArea(max_contour):
            max_contour = contour

    return max_contour


# Calcula los momentos de Hu en base a un contorno
def get_hu_moments(contour):
    moments = cv2.moments(contour)
    return cv2.HuMoments(moments)


def print_hu_moments(greatest_contour, saved_contours):
    if greatest_contour is not None and len(saved_contours) > 0:
        greatest_hu_moments = get_hu_moments(greatest_contour)
        saved_shape_hu_moments = get_hu_moments(saved_contours)

        print('Greatest Contour Hu Moments:\n{}'.format(greatest_hu_moments))
        print('Saved Contours Hu Moments:\n{}'.format(saved_shape_hu_moments))
    else:
        print("No se tiene un contorno guardado actualmente, para guardar uno presione F")


def shape_detector():
    cap = cv2.VideoCapture(0)
    saved_shape = []

    while True:
        _, frame = cap.read()

        # 1. Convertir la imagen a monocromática
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Monocromatica', cv2.flip(gray, 1))

        # 2. Aplicar un threshold con umbral ajustable con una barra de desplazamiento
        #   se pueden incluir opciones de ajuste automático
        trackbar_val = cv2.getTrackbarPos('Trackbar', window_name)
        _, thresh = cv2.threshold(gray, trackbar_val, 255, cv2.THRESH_BINARY_INV)
        # cv2.imshow(window_name, cv2.flip(thresh, 1))

        # 3. Aplicar operaciones morfológicas para eliminar ruido de la imagen
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow(window_name, cv2.flip(closing, 1))

        # 4. Si se obtienen varios contornos, elegir con algún criterio al menos uno para procesar
        #   es preferible procesar todos los contornos significativos, evitando contornos provenientes de ruido
        contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Esto lo agregue para que aparezcan los contornos rojos en la imagen b&n
        new_frame = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)


        # 5. Obtener parámetros del contorno
        #   invariantes de Hu
        #   otros momentos, como m00
        #   otros parámetros relevantes, como concavidades
        if len(contours) > 0:
            greatest_contour = get_greatest_contour(contours)

            sensibility_trackbar_val = cv2.getTrackbarPos('Sensibility trackbar', window_name)

            # Se muestra el contorno mas grande en color azul para diferenciar del resto con un grosor mayor.
            cv2.drawContours(new_frame, [greatest_contour], -1, (255, 0, 0), 12)

            # Con el comando F se guarda el contorno de mayor area.
            if cv2.waitKey(1) & 0xFF == ord('f'):
                saved_shape = greatest_contour

            # Con el comando P se printean los valores de Hu para el contorno de mayor area comparado con el del contorno guardado.
            if cv2.waitKey(1) & 0xFF == ord('p'):
                print_hu_moments(greatest_contour, saved_shape)

            # 6. Clasificación binaria: Aplicar un criterio sobre los invariantes para decidir si el contorno corresponde o no a la pieza a reconocer
            for contour in contours:
                # compares hu moments from two contours
                # 7. Anotar los contornos sobre la imagen monocromática y visualizarla
                if len(saved_shape) > 0 and cv2.matchShapes(contour, saved_shape, cv2.CONTOURS_MATCH_I2, 0) < (sensibility_trackbar_val/100):
                    # Todos los contornos cuyos momentos de Hu sean considerados similares a el contorno guardado se muestran en color verde.
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
                else:
                    # Todos los demas contornos se muestran con un contorno en rojo.
                    cv2.drawContours(frame, [contour], -1, (0, 0, 255), 3)

        cv2.imshow(window_name, cv2.flip(frame, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


shape_detector()
