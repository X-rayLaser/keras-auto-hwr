import numpy as np
from keras.preprocessing.image import array_to_img
from PIL import Image, ImageDraw


def points_to_image(points, width=3):
    height = max([y for x, y in points]) + 1
    width = max([x for x, y in points]) + 1

    a = np.zeros((height, width), dtype=np.uint8)

    im = Image.fromarray(a, mode='L')

    canvas = ImageDraw.ImageDraw(im, mode='L')

    prev_point = None
    for x, y in points:
        if prev_point:
            canvas.line((prev_point, (x, y)), width=width, fill=255)
        prev_point = (x, y)

    return im
