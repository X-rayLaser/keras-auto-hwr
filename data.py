import numpy as np
from keras.utils import to_categorical

from sources.preloaded import PreLoadedSource
from preprocessing import PreProcessor


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

        for points, transcription in self._iter.get_sequences():
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
        return PreLoadedSource(hand_writings, transcriptions)

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
            for handwriting, transcription in self._iter.get_sequences():
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
    def __init__(self, data_source, char_table, num_examples=2000):
        it = self._preload(data_source, num_examples)
        it = self._adapt(it)
        self._char_table = char_table
        self._splitter = DataSplitter(it)

        train_iter = self._splitter.train_data()

        preprocessor = PreProcessor(self._char_table)
        preprocessor.fit(train_iter)

        self._train_iter = preprocessor.process(train_iter)
        self._val_iter = preprocessor.process(self._splitter.validation_data())
        self._test_iter = preprocessor.process(self._splitter.test_data())

    def _adapt(self, source):
        points_seq = []
        transcriptions = []
        for strokes, transcription in source.get_sequences():
            points = []
            for stroke in strokes:
                points.extend(stroke.points)

            points_seq.append(points)
            transcriptions.append(transcription)

        return PreLoadedSource(points_seq, transcriptions)

    def _preload(self, data_source, num_examples):
        hand_writings = []
        transcriptions = []
        for i, (points, t) in enumerate(data_source.get_sequences()):
            if i > num_examples:
                break

            hand_writings.append(points)
            transcriptions.append(t)

        return PreLoadedSource(hand_writings, transcriptions)

    def training_generator(self):
        return DataSetGenerator(self._train_iter, self._char_table)

    def validation_generator(self):
        return DataSetGenerator(self._val_iter,
                                self._char_table)

    def test_generator(self):
        return DataSetGenerator(self._test_iter,
                                self._char_table)
