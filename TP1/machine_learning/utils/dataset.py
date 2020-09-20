import glob
import math

import cv2
import csv

import numpy
import numpy as np
from TP1.common.common_utils import get_greatest_contour

labels = {
    "rectangle": 1,
    "star": 2,
    "triangle": 3,
}


# Genera los momentos de Hu dado un filename, en este caso particular nuestro filename es igual a nuestro label.
def generate_hu_moments_for_shape(filename, writer):
    pathname = '../dataset/images/{}/*.png'.format(filename)
    for file in glob.glob(pathname):
        moments = hu_moments_from_image(file)
        flat = moments.ravel()
        row = np.append(flat, filename)
        write_hu_moments_row(row, writer)


# Calcula los momentos de Hu en base a una imagen pasada como parametro
def hu_moments_from_image(file):
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    block_size = 67 #Tamaño del bloque a comparar, debe ser impar.
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2)

    # Invert the image so the area of the UAV is filled with 1's. This is necessary since
    # cv::findContours describes the boundary of areas consisting of 1's.
    bin = 255 - bin  # como sabemos que las figuras son negras invertimos los valores binarios para que esten en 1.

    # Tamaño del bloque a recorrer
    kernel = numpy.ones((3, 3), numpy.uint8)
    # buscamos eliminar falsos positivos (puntos blancos en el fondo) para eliminar ruido.
    bin = cv2.morphologyEx(bin, cv2.MORPH_ERODE, kernel)

    # encuetra los contornos, chain aprox simple une algunos puntos para que no sea discontinuo.
    contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # Agarra el contorno de area maxima
    shape_contour = max(contours, key=cv2.contourArea)

    moments = cv2.moments(shape_contour)  # momentos de inercia
    huMoments = cv2.HuMoments(moments)  # momentos de Hu

    # hacemos esto para que los valores no sean taaan chiquitos (log)
    for i in range(0, 7):
        huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(
            abs(huMoments[i]))  # Mapeo para agrandar la escala.
    return huMoments

def write_hu_moments_row(row, writer):
    writer.writerow(row)


def generate_hu_moments():
    with open('../dataset/moments.csv', mode='w', newline='') as moments_file:
        writer = csv.writer(moments_file)

        generate_hu_moments_for_shape('rectangle', writer)
        generate_hu_moments_for_shape('star', writer)
        generate_hu_moments_for_shape('triangle', writer)


# Convierto un label de interes en un valor entero para el entrenamiento.
def label_to_int(label):
    global labels
    return int(labels.get(label))


# Convierto de un valor entero al label que me interesa, devuelvo None si no existe.
def int_to_label(search):
    global labels
    for key, value in labels.items():
        if value == search:
            return key

    return None


if __name__ == '__main__':
    generate_hu_moments()
