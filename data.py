import xml.etree.ElementTree as ET
from PIL.ImageDraw import ImageDraw
from PIL import Image, ImageFont
import numpy as np
import os
import random
from keras.preprocessing.image import array_to_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


class CharacterTable:
    def __init__(self, corpus):
        charset = set(corpus + self.sentinel + self.start)
        self._char_to_index = dict((ch, index) for index, ch in enumerate(charset))
        self._index_to_char = dict((v, k) for k, v in self._char_to_index.items())

    def save(self, path):
        import json
        d = {
            'char_to_index': self._char_to_index,
            'index_to_char': self._index_to_char
        }
        json_str = json.dumps(d)
        with open(path, 'w') as f:
            f.write(json_str)

    def load(self, path):
        import json
        with open(path, 'r') as f:
            s = f.read()

        d = json.loads(s)
        self._char_to_index = d['char_to_index']
        self._index_to_char = d['index_to_char']

        d = {}
        for k, v in self._index_to_char.items():
            d[int(k)] = v

        self._index_to_char = d

    def is_sentinel(self, ch):
        return self.sentinel == ch

    @property
    def sentinel(self):
        return '\n'

    @property
    def start(self):
        return '_'

    def encode(self, ch):
        return self._char_to_index[ch]

    def decode(self, index):
        if index in self._index_to_char:
            return self._index_to_char[index]
        else:
            raise Exception('Unknown {}'.format(index))
            return '?'

    def __len__(self):
        return len(self._char_to_index)


class StrokesNotFoundException(Exception):
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
    def __init__(self, path):
        tree = ET.parse(path)
        root = tree.getroot()

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


class BaseIterator:
    def get_lines(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SyntheticIterator(BaseIterator):
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

    def get_lines(self):

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


class RawIterator(BaseIterator):
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

    def get_strokes_path(self, strokes_root, file_id):
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

        return stroke_path

    def get_transcriptions(self):
        for transcription_path in self.transcription_paths():
            try:
                transcription = Transcription(transcription_path)
                yield transcription
            except MissingTranscriptionException:
                continue

    def get_lines(self):
        strokes_root = os.path.join(self._root, 'lineStrokes-all', 'lineStrokes')

        for transcription in self.get_transcriptions():
            for file_id, true_text in transcription.text_lines():
                stroke_path = self.get_strokes_path(strokes_root, file_id)

                try:
                    stroke_line = StrokeLine(stroke_path)
                except StrokesNotFoundException:
                    continue

                line_points = stroke_line.points()

                yield line_points, true_text

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


class PreLoadedIterator(BaseIterator):
    def __init__(self, hand_writings, transcriptions):
        self._num_examples = len(transcriptions)
        self._line_points = hand_writings
        self._transcriptions = transcriptions

    def __len__(self):
        return len(self._transcriptions)

    def get_lines(self):
        indices = list(range(len(self._transcriptions)))

        random.shuffle(indices)

        for i in indices:
            yield self._line_points[i], self._transcriptions[i]


class DataSplitter:
    def __init__(self, data_iterator):
        self._iter = data_iterator

        self._train = []
        self._val = []
        self._test = []
        self._split()

    def _split(self):
        destination = (self._train, self._val, self._test)

        pmf = [0.9, 0.05, 0.05]

        for points, transcription in self._iter.get_lines():
            index = np.random.choice([0, 1, 2], replace=True, p=pmf)
            dest_list = destination[index]
            dest_list.append((points, transcription))

        if len(self._val) == 0:
            self._val.append(self._train.pop())

        if len(self._test) == 0:
            self._test.append(self._train.pop())

        assert len(self._train) > 0

    def _create_iterator(self, data):
        hand_writings = [points for points, transcription in data]
        transcriptions = [transcription for points, transcription in data]
        return PreLoadedIterator(hand_writings, transcriptions)

    def train_data(self):
        return self._create_iterator(self._train)

    def validation_data(self):
        return self._create_iterator(self._val)

    def test_data(self):
        return self._create_iterator(self._test)


class DataSetGenerator:
    def __init__(self, lines_iterator, char_table):
        self._char_table = char_table
        self._iter = lines_iterator

    def prepare_batch(self, hand_writings, transcriptions):
        x = np.array(hand_writings)

        x_norm = x.reshape((-1, x.shape[1], 1))

        char_table = self._char_table
        seqs = []

        for tok in transcriptions:
            s = char_table.start + tok + char_table.sentinel
            encoded = [char_table.encode(ch) for ch in s]
            seqs.append(encoded)

        padded_seqs = np.array(seqs)

        y = to_categorical(padded_seqs, num_classes=len(char_table))
        y_in = y[:, :-1, :]
        y_out = y[:, 1:, :]

        return [x_norm, y_in], y_out

    def get_examples(self, batch_size=64):
        while True:
            hand_writings = []
            transcriptions = []
            for handwriting, transcription in self._iter.get_lines():
                hand_writings.append(handwriting)
                transcriptions.append(transcription)

                if len(hand_writings) >= batch_size:
                    yield self.prepare_batch(hand_writings, transcriptions)

                    hand_writings = []
                    transcriptions = []

            if len(hand_writings) > 0:
                yield self.prepare_batch(hand_writings, transcriptions)

        # todo: separate generator for images instead of strokes

    def __len__(self):
        return len(self._iter)


class DataFactory:
    def __init__(self, data_root, char_table, num_examples=2000):
        it = self._preload(data_root, num_examples)
        self._char_table = char_table
        self._splitter = DataSplitter(it)

        train_iter = self._splitter.train_data()

        from preprocessing import PreProcessor

        preprocessor = PreProcessor()
        preprocessor.fit(train_iter)

        self._train_iter = preprocessor.process(train_iter)
        self._val_iter = preprocessor.process(self._splitter.validation_data())
        self._test_iter = preprocessor.process(self._splitter.test_data())

    def _get_iterator(self, data_root, num_examples):
        return RandomOrderIterator(data_root)

    def _preload(self, data_root, num_examples):
        rnd_iterator = self._get_iterator(data_root, num_examples)

        hand_writings = []
        transcriptions = []
        for i, (points, t) in enumerate(rnd_iterator.get_lines()):
            if i > num_examples:
                break

            hand_writings.append(points)
            transcriptions.append(t)

        return PreLoadedIterator(hand_writings, transcriptions)

    def training_generator(self):
        return DataSetGenerator(self._train_iter, self._char_table)

    def validation_generator(self):
        return DataSetGenerator(self._val_iter,
                                self._char_table)

    def test_generator(self):
        return DataSetGenerator(self._test_iter,
                                self._char_table)


class SyntheticFactory(DataFactory):
    def _get_iterator(self, data_root, num_examples):
        return SyntheticIterator(num_examples)
