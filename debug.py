from data.char_table import CharacterTable
from data.preprocessing import PreProcessor, Flattening, SignalMaker, DeltaSignal
from sources.compiled import CompilationSource
import numpy as np
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt


charset = ''.join([chr(i) for i in range(32, 128)])
char_table = CharacterTable(charset)

source = CompilationSource('./compiled/data.json')


def visualize_strokes_input(strokes, output_dir='./debug_output'):
    points = []

    for stroke in strokes:
        for point in stroke.points:
            points.append(point)

    width = max([x for x, y in points])
    height = max([y for x, y in points])

    a = np.zeros((height, width), dtype=np.uint8)

    im = Image.fromarray(a, mode='L')

    canvas = ImageDraw.ImageDraw(im, mode='L')

    prev_point = None
    for i, (x, y) in enumerate(points):
        if prev_point:
            canvas.line((prev_point, (x, y)), width=4, fill=255)

            save_path = os.path.join(output_dir, '{}.jpg'.format(i))
            im.save(save_path)
        prev_point = (x, y)


def visualize_strokes(strokes):
    for stroke in strokes:
        print('Points in stroke', len(stroke.points))
        max_x = max([x for x, y in stroke.points])
        min_x = min([x for x, y in stroke.points])

        width = max_x - min_x

        max_y = max([y for x, y in stroke.points])
        min_y = min([y for x, y in stroke.points])
        height = max_y - min_y

        a = np.zeros((height, width), dtype=np.uint8)

        im = Image.fromarray(a, mode='L')

        canvas = ImageDraw.ImageDraw(im, mode='L')
        prev_point = None
        for i, (x, y) in enumerate(stroke.points):
            x = x - min_x
            y = y - min_y
            if prev_point:
                canvas.line((prev_point, (x, y)), width=4, fill=255)
            prev_point = (x, y)
        #im.show()
        #input('Press any key')


def visualize_signal(signal):
    indices = list(range(len(signal)))
    plt.scatter(indices, signal)
    plt.show()


pre_processor1 = PreProcessor('')
pre_processor1.add_step(Flattening())

pre_processor2 = PreProcessor('')
pre_processor2.add_step(Flattening())
pre_processor2.add_step(SignalMaker())

pre_processor3 = PreProcessor('')
pre_processor3.add_step(Flattening())
pre_processor3.add_step(SignalMaker())
pre_processor3.add_step(DeltaSignal())

in_seqs = []
out_seqs = []
for in_seq, out_seq in source.get_sequences():
    in_seqs.append(in_seq)
    out_seqs.append(out_seq)
    break
