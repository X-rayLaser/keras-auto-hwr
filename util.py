import numpy as np
from keras.preprocessing.image import array_to_img
from PIL import Image, ImageDraw


def round_point(point):
    x, y = point
    return int(round(x)), int(round(y))


def scale(points, factor=100):
    return [(x * factor, y * factor) for x, y in points]


def decode(deltas):
    points = []
    for grad in deltas:
        dx, dy = grad

        if len(points) == 0:
            prev_point = (0, 0)
        else:
            prev_point = points[-1]
        x = prev_point[0] + dx
        y = prev_point[1] + dy
        points.append((x, y))

    min_x = min([x for x, y in points])
    min_y = min([y for x, y in points])

    res = []
    for x, y in points:
        res.append((x - min_x, y - min_y))

    return res


def decode_alphas(signal):
    points = [(0, 0)]
    prev_alpha = 0
    import math
    for delta_alpha, dist in signal:
        prev_alpha += delta_alpha
        x, y = points[-1]
        dx = dist * math.cos(prev_alpha)
        dy = dist * math.sin(prev_alpha)
        points.append((x + dx, y + dy))

    min_x = min([x for x, y in points])
    min_y = min([y for x, y in points])

    res = []
    for x, y in points:
        res.append((x - min_x, y - min_y))

    return res


def visualize_stroke(points):
    points = decode(points)
    #points = decode_alphas(points)
    points = scale(points)
    points = [round_point(p) for p in points]
    print('Points in stroke', len(points))
    import numpy as np
    from PIL import Image
    from PIL import ImageDraw
    max_x = max([x for x, y in points])

    width = max_x + 1

    max_y = max([y for x, y in points])
    height = max_y + 1

    a = np.zeros((height, width), dtype=np.uint8)

    im = Image.fromarray(a, mode='L')

    canvas = ImageDraw.ImageDraw(im, mode='L')
    prev_point = None
    for i, (x, y) in enumerate(points):
        #x = x - min_x
        #y = y - min_y
        if prev_point:
            canvas.line((prev_point, (x, y)), width=4, fill=255)
        prev_point = (x, y)

    return im


def points_to_image(points, width=3):
    #points = [round_point(p) for p in points]
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
