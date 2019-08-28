import numpy as np
from keras.utils import to_categorical

from sources.preloaded import PreLoadedSource
from preprocessing import PreProcessor
from util import points_to_image
from urllib.parse import quote
import os


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


class AttentionModelDataGenerator(DataSetGenerator):
    def __init__(self, lines_iterator, char_table, Tx, Ty, encoder_states):
        super().__init__(lines_iterator, char_table)

        self._Tx = Tx
        self._Ty = Ty
        self._encoder_states = encoder_states

    def prepare_batch(self, hand_writings, transcriptions):
        batch_size = len(transcriptions)
        alphabet_size = len(self._char_table)

        x = np.array(hand_writings)
        x_norm = x.reshape((-1, x.shape[1], 1))
        initial_state = np.zeros((batch_size, self._encoder_states))

        targets = []
        for tok in transcriptions:
            s = tok + self._char_table.sentinel

            encoded = [self._char_table.encode(ch) for ch in s]
            targets.append(encoded)

        targets = np.array(targets)
        y = to_categorical(targets, num_classes=alphabet_size)

        final_y = []
        for t in range(self._Ty):
            final_y.append(y[:, t, :])

        initial_y = np.zeros((batch_size, 1, alphabet_size))
        return [x_norm, initial_state, initial_y], final_y


class BaseFactory:
    def __init__(self, data_source, char_table, num_examples=2000, save_images_path=None):
        self._data_source = data_source

        self._char_table = char_table
        self._num_examples = num_examples
        self._save_images_path = save_images_path

        self._train_iter = None
        self._val_iter = None
        self._test_iter = None

    def create_model(self):
        raise NotImplementedError

    def training_generator(self):
        raise NotImplementedError

    def validation_generator(self):
        raise NotImplementedError

    def test_generator(self):
        raise NotImplementedError

    def prepare_sources(self):
        it = self._preload()
        it = self._adapt(it)
        splitter = DataSplitter(it)

        train_iter = splitter.train_data()

        preprocessor = self._get_preprocessor()
        preprocessor.fit(train_iter)

        self._train_iter = preprocessor.process(train_iter)
        self._val_iter = preprocessor.process(splitter.validation_data())
        self._test_iter = preprocessor.process(splitter.test_data())

    def _get_preprocessor(self):
        raise NotImplementedError

    def save_points(self, points, transcription):
        file_name = '{}.jpg'.format(quote(transcription, safe=' ,.'))
        path = os.path.join(self._save_images_path, file_name)

        points_to_image(points).save(path)

    def _adapt(self, source):
        points_seq = []
        transcriptions = []
        for strokes, transcription in source.get_sequences():
            points = []
            for stroke in strokes:
                points.extend(stroke.points)

            points_seq.append(points)
            transcriptions.append(transcription)

            if self._save_images_path:
                self.save_points(points, transcription)

        return PreLoadedSource(points_seq, transcriptions)

    def _preload(self):
        hand_writings = []
        transcriptions = []
        for i, (points, t) in enumerate(self._data_source.get_sequences()):
            if i > self._num_examples:
                break

            hand_writings.append(points)
            transcriptions.append(t)

        return PreLoadedSource(hand_writings, transcriptions)


from preprocessing import SignalMaker, DeltaSignal, SequencePadding, Normalization


class Seq2seqFactory(BaseFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_preprocessor(self):
        preprocessor = PreProcessor(self._char_table)
        preprocessor.add_step(SignalMaker())
        preprocessor.add_step(DeltaSignal())
        return preprocessor

    def training_generator(self):
        return DataSetGenerator(self._train_iter, self._char_table)

    def validation_generator(self):
        return DataSetGenerator(self._val_iter,
                                self._char_table)

    def test_generator(self):
        return DataSetGenerator(self._test_iter,
                                self._char_table)

    def create_model(self):
        from models.seq2seq import SequenceToSequenceTrainer
        return SequenceToSequenceTrainer(self._char_table)


class AttentionalSeq2seqFactory(BaseFactory):
    def __init__(self, Tx, Ty, num_cells, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._Tx = Tx
        self._Ty = Ty
        self._num_cells = num_cells

    def _get_preprocessor(self):
        preprocessor = PreProcessor(self._char_table)
        preprocessor.add_step(SignalMaker())
        preprocessor.add_step(DeltaSignal())
        preprocessor.add_step(
            SequencePadding(target_padding=self._char_table.sentinel,
                            input_len=self._Tx, output_len=self._Ty - 1)
        )

        preprocessor.add_step(Normalization())
        return preprocessor

    def create_model(self):
        from models.attention import Seq2SeqWithAttention
        return Seq2SeqWithAttention(self._char_table, self._num_cells,
                                    Tx=self._Tx, Ty=self._Ty)

    def _get_generator(self, iterator):
        return AttentionModelDataGenerator(iterator, self._char_table, self._Tx, self._Ty, self._num_cells)

    def training_generator(self):
        return self._get_generator(self._train_iter)

    def validation_generator(self):
        return self._get_generator(self._val_iter)

    def test_generator(self):
        return self._get_generator(self._test_iter)
