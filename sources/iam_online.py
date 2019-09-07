import os
import random
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image
from PIL.ImageDraw import ImageDraw

from sources import BaseSource


class OnlineSource(BaseSource):
    def __init__(self, data_root):
        self._root = data_root
        self._strokes_root = os.path.join(self._root, 'lineStrokes-all',
                                          'lineStrokes')

    def transcription_paths(self):
        labels_root = os.path.join(self._root, 'original-xml-all', 'original')

        for dirname1 in os.listdir(labels_root):
            path1 = os.path.join(labels_root, dirname1)
            for dirname2 in os.listdir(path1):
                path2 = os.path.join(path1, dirname2)
                for transcription_file in os.listdir(path2):
                    transcription_path = os.path.join(path2, transcription_file)
                    yield transcription_path

    def get_strokes_path(self, file_id):
        path_components = file_id.split('-')
        if path_components[1][-1].isalpha():
            subfolder = path_components[0] + '-' + path_components[1][:-1]
        else:
            subfolder = path_components[0] + '-' + path_components[1]

        stroke_path = os.path.join(
            self._strokes_root, path_components[0],
            subfolder,
            file_id + '.xml'
        )

        return stroke_path

    def get_transcriptions(self, random_order=False):
        for transcription_path in self.transcription_paths():
            try:
                transcription = Transcription(transcription_path, random_order)
                yield transcription
            except MissingTranscriptionException:
                continue

    def get_sequences(self):
        for transcription in self.get_transcriptions():
            for file_id, true_text in transcription.text_lines():
                stroke_path = self.get_strokes_path(file_id)

                try:
                    stroke_line = StrokeLine(stroke_path)
                except StrokesNotFoundException:
                    continue

                line_strokes = stroke_line.strokes()

                yield line_strokes, true_text

    def __len__(self):
        counter = 0
        for _ in self.get_sequences():
            counter += 1
        return counter


class LinesSource(BaseSource):
    def __init__(self, source):
        self._source = source

    def get_sequences(self):
        for line in self._source.get_sequences():
            yield line

    def __len__(self):
        return len(self._source)


from sources.preloaded import PreLoadedSource


def fetch_strokes(source, num_strokes):
    cropped_strokes = []
    dummy_out = []
    for strokes, text in source.get_sequences():
        for stroke in strokes:
            if len(cropped_strokes) > num_strokes:
                return PreLoadedSource(cropped_strokes, dummy_out)

            try:
                deltas = stroke.stroke_to_points()
            except BadStrokeException:
                continue

            cropped_strokes.append(deltas)
            dummy_out.append('')

    return PreLoadedSource(cropped_strokes, dummy_out)


class StrokesSource(BaseSource):
    def __init__(self, source, num_strokes):
        self._source = source
        self._num_strokes = num_strokes
        self._preloaded = fetch_strokes(source, num_strokes)

    def get_sequences(self):
        for line in self._preloaded.get_sequences():
            yield line

    def __len__(self):
        return self._num_strokes


class WordsSource(BaseSource):
    def __init__(self, source):
        self._source = source

    def _distance_deviation(self, strokes):
        delayed_strokes = strokes[1:]

        distances = []
        for i in range(len(delayed_strokes)):
            next_stroke = delayed_strokes[i]
            stroke = strokes[i]
            distances.append(next_stroke.horizontal_distance(stroke))

        return np.std(distances)

    def _word_strokes(self, strokes):
        if len(strokes) == 0:
            return
        sd = self._distance_deviation(strokes)

        word_strokes = []

        prev_stroke = strokes[0]
        word_strokes.append(prev_stroke)
        for stroke in strokes[1:]:
            if stroke.horizontal_distance(prev_stroke) > 1.5 * sd:
                yield word_strokes
                return
                word_strokes = []
            word_strokes.append(stroke)
            prev_stroke = stroke

    def get_sequences(self):
        for strokes, transcription in self._source.get_sequences():
            word_transcriptions = transcription.split(' ')

            for i, word_strokes in enumerate(self._word_strokes(strokes)):
                if i < len(word_transcriptions):
                    yield word_strokes, word_transcriptions[i]

    def __len__(self):
        pass


class RandomOrderSource(OnlineSource):
    def transcription_paths(self):
        all_paths = []
        for path in super().transcription_paths():
            all_paths.append(path)

        random.shuffle(all_paths)

        for path in all_paths:
            yield path

    def make_generator(self, transcription_path):
        try:
            transcription = Transcription(transcription_path, random_order=True)
            for line in transcription.text_lines():
                yield line
        except MissingTranscriptionException:
            pass

    def get_sequences(self):
        line_iterators = []

        for path in self.transcription_paths():
            line_iterators.append(self.make_generator(path))

        while len(line_iterators) > 0:
            gen = random.choice(line_iterators)

            try:
                file_id, true_text = next(gen)
            except StopIteration:
                line_iterators.remove(gen)
                continue

            stroke_path = self.get_strokes_path(file_id)

            try:
                stroke_line = StrokeLine(stroke_path)
            except StrokesNotFoundException:
                continue

            line_points = stroke_line.points()

            yield line_points, true_text


class StrokesNotFoundException(Exception):
    pass


class Stroke:
    def __init__(self, points):
        self.points = points

    def left_most_x(self):
        return min([x for x, y in self.points])

    def right_most_x(self):
        return min([x for x, y in self.points])

    def horizontal_distance(self, stroke):
        return self.left_most_x() - stroke.right_most_x()

    def stroke_to_points(self):
        max_x = max([x for x, y in self.points])
        min_x = min([x for x, y in self.points])

        max_y = max([y for x, y in self.points])
        min_y = min([y for x, y in self.points])

        width = max_x - min_x
        height = max_y - min_y

        max_side = max(width, height) + 0.00001
        if max_side < 1:
            raise BadStrokeException()

        points = [(0, 0)]
        for x, y in self.points:
            x = (x - min_x) / max_side
            y = (y - min_y) / max_side
            points.append((x, y))

        prev_point = (0, 0)
        deltas = []

        for p in points:
            x, y = p
            dx = x - prev_point[0]
            dy = y - prev_point[1]

            deltas.append((dx, dy))

            prev_point = p

        points.append((0, 0))

        deltas.append((0, 0))

        return deltas


class BadStrokeException(Exception):
    pass


class StrokeLine:
    def __init__(self, xml_path):
        if not os.path.isfile(xml_path):
            raise StrokesNotFoundException()

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
        return [(x, y) for x, y, prev in self.points_generator()]

    def strokes(self):
        stroke_set = list(self._root.iterfind('StrokeSet'))[0]

        res = []

        for stroke_tag in stroke_set:
            assert stroke_tag.attrib['colour'] == 'black'

            points = []

            for point in stroke_tag:
                y = int(point.attrib['y']) - self._y_from
                x = int(point.attrib['x']) - self._x_from
                points.append((x, y))

            res.append(Stroke(points))

        return res

    def heights(self):
        v = []

        for x, y, prev in self.points_generator():
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
    def __init__(self, path, random_order=False):
        tree = ET.parse(path)
        root = tree.getroot()

        transcription_tag = list(root.iterfind('Transcription'))
        if len(transcription_tag) == 0:
            raise MissingTranscriptionException()

        self._transcription = transcription_tag[0]

        self._random_order = random_order
        self._path = path
        self._lines = None

    def _fetch_lines(self, ):
        if self._lines is None:
            self._lines = []
            for line in self._transcription.iterfind('TextLine'):
                text = line.attrib['text']
                file_id = line.attrib['id']

                second_opinion = self.id_from_words(line)
                if file_id != second_opinion:
                    print('ATTENTION: id attribute "file_id" on line does not match the one on words: {} vs {}'.format(file_id, second_opinion))

                self._lines.append((second_opinion, text))

            if self._random_order:
                random.shuffle(self._lines)

        return self._lines

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
        for line in self._fetch_lines():
            yield line


class MissingTranscriptionException(Exception):
    pass