import random

import numpy as np
from PIL import ImageFont, Image
from PIL.ImageDraw import ImageDraw
from keras.preprocessing.image import img_to_array, array_to_img

from sources import BaseSource


class SyntheticIterator(BaseSource):
    def __init__(self, num_lines):
        self._num_lines = num_lines

    def _create_image(self, text):
        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)

        h = 40
        w = int(round(2/3 * h * len(text)))
        a = np.zeros((h, w), dtype=np.uint8)

        im = Image.fromarray(a, mode='L')

        canvas = ImageDraw(im, mode='L')
        canvas.text((0, 0), text, fill=255, font=fnt)

        return im

    def _to_point_sequence(self, image):
        a = img_to_array(image)

        h, w, _ = a.shape
        a = a.reshape(h, w)

        dummy = np.arange(w)

        index_tuples = []
        for i, row in enumerate(a):
            columns = dummy[row > 0]
            for col in columns:
                index_tuples.append((col, i))

        return sorted(index_tuples, key=lambda t: (t[0], t[1]))

    def _points_to_image(self, points):
        height = max([y for x, y in points]) + 1
        width = max([x for x, y in points]) + 1

        a = np.zeros((height, width), dtype=np.uint8)

        cols = [x for x, y in points]
        rows = [y for x, y in points]
        a[rows, cols] = 255

        return array_to_img(a.reshape((height, width, 1)))

    def get_sequences(self):

        words = ['world', 'travel', 'book', 'ticket', 'take', 'word', 'titan']

        counter = 0
        while True:
            random.shuffle(words)

            for word in words:
                im = self._create_image(word)

                points = self._to_point_sequence(im)

                yield points, word

                counter += 1

                if counter >= self._num_lines:
                    return

    def __len__(self):
        return self._num_lines


# todo: fix get_lines method
