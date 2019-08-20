import xml.etree.ElementTree as ET
from PIL.ImageDraw import ImageDraw
from PIL import Image
import numpy as np
import os
import random


class StrokeLine:
    def __init__(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        board_info = list(root.iterfind('WhiteboardDescription'))[0]
        location = list(board_info.iterfind('SensorLocation'))[0]
        assert location.attrib['corner'] == 'top_left'

        diagonally_opposite = list(board_info.iterfind('DiagonallyOppositeCoords'))[0]
        vertically_opposite = list(board_info.iterfind('VerticallyOppositeCoords'))[0]
        horizontally_opposite = list(board_info.iterfind('HorizontallyOppositeCoords'))[0]

        x1 = int(diagonally_opposite.attrib['x'])
        y1 = int(diagonally_opposite.attrib['y'])
        x2 = int(vertically_opposite.attrib['x'])
        y2 = int(vertically_opposite.attrib['y'])
        x3 = int(horizontally_opposite.attrib['x'])
        y3 = int(horizontally_opposite.attrib['y'])

        x_from = min(x1, x2, x3)
        x_to = max(x1, x2, x3)

        y_from = min(y1, y2, y3)
        y_to = max(y1, y2, y3)

        height = y_to - y_from + 1
        width = x_to - x_from + 1

        self._x_from = x_from
        self._y_from = y_from
        self._x_max = x_to
        self._y_max = y_to
        self._height = height
        self._width = width

        self._root = root

    def points_generator(self):
        stroke_set = list(self._root.iterfind('StrokeSet'))[0]

        for stroke in stroke_set:
            assert stroke.attrib['colour'] == 'black'

            prev_point = None

            for point in stroke:
                y = int(point.attrib['y']) - self._y_from
                x = int(point.attrib['x']) - self._x_from
                yield x, y, prev_point
                prev_point = (x, y)

    def points(self):
        points = [(x, y) for x, y, prev in self.points_generator()]
        v = []
        for x, y in points:
            v.append(x)
            v.append(y)

        return v

    def to_image(self):
        a = np.zeros((self._height, self._width), dtype=np.uint8)

        im = Image.fromarray(a, mode='L')

        canvas = ImageDraw(im, mode='L')

        for x, y, prev_point in self.points_generator():
            if prev_point:
                canvas.line((prev_point, (x, y)), width=12, fill=255)

        return im


class Transcription:
    def __init__(self, path):

        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except:
            print('INTERESTING ERROR')
            import traceback
            traceback.print_exc()
            raise Exception()

        transcription_tag = list(root.iterfind('Transcription'))
        if len(transcription_tag) == 0:
            raise MissingTranscriptionException()

        self._transcription = transcription_tag[0]

    def id_from_words(self, line):
        id_counts = {}
        for word in line.iterfind('Word'):
            word_id = word.attrib['id']
            components = word_id.split('-')
            line_id = '-'.join(components[:3])

            if line_id not in id_counts:
                id_counts[line_id] = 0
            id_counts[line_id] += 1

        res = None
        max_count = 0
        for line_id, count in id_counts.items():
            if count > max_count:
                max_count = count
                res = line_id

        return res

    def text_lines(self):
        for line in self._transcription.iterfind('TextLine'):
            text = line.attrib['text']
            file_id = line.attrib['id']

            second_opinion = self.id_from_words(line)
            if file_id != second_opinion:
                print('ATTENTION: id attribute "file_id" on line does not match the one on words: {} vs {}'.format(file_id, second_opinion))

            yield second_opinion, text


class MissingTranscriptionException(Exception):
    pass


class RawIterator:
    def __init__(self, data_root):
        self._root = data_root

    def transcription_paths(self):
        labels_root = os.path.join(self._root, 'original-xml-all', 'original')

        for dirname1 in os.listdir(labels_root):
            path1 = os.path.join(labels_root, dirname1)
            for dirname2 in os.listdir(path1):
                path2 = os.path.join(path1, dirname2)
                for transcription_file in os.listdir(path2):
                    transcription_path = os.path.join(path2, transcription_file)
                    yield transcription_path

    def get_lines(self):
        strokes_root = os.path.join(self._root, 'lineStrokes-all', 'lineStrokes')

        for transcription_path in self.transcription_paths():
            try:
                transcription = Transcription(transcription_path)
            except:
                continue

            for file_id, true_text in transcription.text_lines():
                path_components = file_id.split('-')
                if path_components[1][-1].isalpha():
                    subfolder = path_components[0] + '-' + path_components[1][:-1]
                else:
                    subfolder = path_components[0] + '-' + path_components[1]

                stroke_path = os.path.join(
                    strokes_root, path_components[0],
                    subfolder,
                    file_id + '.xml'
                )

                try:
                    stroke_line = StrokeLine(stroke_path)
                    line_points = stroke_line.points()
                    yield line_points, true_text
                except:
                    import traceback
                    traceback.print_exc()

    def __len__(self):
        counter = 0
        for _ in self.get_lines():
            counter += 1
        return counter


class RandomOrderIterator(RawIterator):
    def transcription_paths(self):
        labels_root = os.path.join(self._root, 'original-xml-all', 'original')

        dirs = os.listdir(labels_root)

        random.shuffle(dirs)

        for d in dirs:
            dir_path = os.path.join(labels_root, d)

            sub_dirs = os.listdir(dir_path)

            random.shuffle(sub_dirs)

            for sub_dir in sub_dirs:
                sub_dir_path = os.path.join(dir_path, sub_dir)

                transcription_files = os.listdir(sub_dir_path)

                random.shuffle(transcription_files)

                for file_name in transcription_files:
                    transcription_path = os.path.join(sub_dir_path, file_name)
                    yield transcription_path
