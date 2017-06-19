import numpy as np
import cv2
from PIL import Image


def process_image(image):
    im_arr = np.array(image).astype(np.uint8)
    im = cv2.cvtColor(im_arr, cv2.COLOR_GRAY2BGR)
    mask = np.zeros(image.shape, np.uint8)
    contours, hierarchy = cv2.findContours(im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000000:
            cv2.drawContours(mask, [cnt], 0, 255, -1)

    return mask
