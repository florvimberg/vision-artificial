import cv2


# Da como resultado el contorno con el area maxima dada una lista de contornos.
def get_greatest_contour(contours):
    max_contour = contours[0]
    for contour in contours:
        if cv2.contourArea(contour) > cv2.contourArea(max_contour):
            max_contour = contour

    return max_contour