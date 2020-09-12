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

def shapeDetector():
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        # 1. Convertir la imagen a monocromática
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Monocromatica', cv2.flip(gray, 1))

        # 2. Aplicar un threshold con umbral ajustable con una barra de desplazamiento
        #   se pueden incluir opciones de ajuste automático
        trackbar_val = cv2.getTrackbarPos(trackbar_name, window_name)
        _, thresh = cv2.threshold(gray, trackbar_val, 255, cv2.THRESH_BINARY)
        # cv2.imshow(window_name, cv2.flip(thresh, 1))

        # 3. Aplicar operaciones morfológicas para eliminar ruido de la imagen
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow(window_name, cv2.flip(closing, 1))

        # 4. Si se obtienen varios contornos, elegir con algún criterio al menos uno para procesar
        #   es preferible procesar todos los contornos significativos, evitando contornos provenientes de ruido
        contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # Esto lo agregue para que aparezcan los contornos rojos en la imagen b&n
        new_frame = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(new_frame, contours, -1, (0, 0, 255), 3)

        # 5. Obtener parámetros del contorno
        #   invariantes de Hu
        #   otros momentos, como m00
        #   otros parámetros relevantes, como concavidades
        moments = cv2.moments(contours[1])
        huMoments = cv2.HuMoments(moments)

        cv2.imshow(window_name, cv2.flip(new_frame, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


shapeDetector()