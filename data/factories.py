import os
from urllib.parse import quote

import numpy as np

from data import PreProcessor, DataSetGenerator, AttentionModelDataGenerator
from models.attention import Seq2SeqWithAttention
from sources.preloaded import PreLoadedSource
from util import points_to_image
from . import preprocessing
from config import Config
from sources.compiled import H5pyDataSet
import h5py
from sources.wrappers import H5pySource


class H5pyRank3DataSet(H5pyDataSet):
    @staticmethod
    def create(path):
        super(H5pyRank3DataSet, H5pyRank3DataSet).create(path)
        with h5py.File(path, 'a') as f:
            f.create_group('stroke_lengths')

        return H5pyRank3DataSet(path)

    def add_example(self, strokes, transcription_text):
        m = len(self)

        flatten = []
        stroke_lens = []
        for stroke in strokes:
            flatten.extend(stroke)
            stroke_lens.append(len(stroke))

        super().add_example(flatten, transcription_text)

        with h5py.File(self._path, 'a') as f:
            lens = f['stroke_lengths']
            lens_dset = lens.create_dataset(str(m), data=np.array(stroke_lens))
            lens_dset.flush()

    def get_example(self, index):
        xs, ys = super().get_example(index)
        with h5py.File(self._path, 'r') as f:
            lengths_group = f['stroke_lengths']
            dict_key = str(index)
            lengths = lengths_group[dict_key]

            strokes = []

            index = 0
            for stroke_length in lengths:
                stroke = xs[index:index + stroke_length]
                strokes.append(stroke.tolist())
                index += stroke_length

            return strokes, ys


class BaseBuffer:
    def __len__(self):
        raise NotImplementedError

    def pop(self):
        raise NotImplementedError

    def get_data(self, num_lines=None, random_order=False):
        raise NotImplementedError

    def add_example(self, strokes, transcription_text):
        raise NotImplementedError


class H5pyBuffer(BaseBuffer):
    def __init__(self, path):
        self.ds = H5pyRank3DataSet.create(path)

    def add_example(self, strokes, transcription_text):
        self.ds.add_example(strokes, transcription_text)

    def pop(self):
        return self.ds.pop()

    def get_data(self, num_lines=None, random_order=False):
        return self.ds.get_data(random_order=random_order)

    def __len__(self):
        return len(self.ds)


class BaseBufferFactory:
    def get_train_buffer(self):
        raise NotImplementedError

    def get_validation_buffer(self):
        raise NotImplementedError

    def get_test_buffer(self):
        raise NotImplementedError


class H5pyBufferFactory(BaseBufferFactory):
    def __init__(self, location):
        self.location = location
        self.train_path = os.path.join(location, 'train.h5py')
        self.val_path = os.path.join(location, 'validation.h5py')
        self.test_path = os.path.join(location, 'test.h5py')

    def get_buffer(self, path):
        return H5pyBuffer(path)

    def get_train_buffer(self):
        return self.get_buffer(self.train_path)

    def get_validation_buffer(self):
        return self.get_buffer(self.val_path)

    def get_test_buffer(self):
        return self.get_buffer(self.test_path)


class DataSplitter:
    @staticmethod
    def validate(source, training_fraction, validation_fraction):
        if training_fraction > 1 or validation_fraction > 1:
            raise BadFractionsException()

        if training_fraction < 0 or validation_fraction < 0:
            raise BadFractionsException()

        if training_fraction + validation_fraction > 1:
            raise BadFractionsException()

        if len(source) < 3:
            raise InsufficientNumberOfExamplesException()

    @classmethod
    def create(cls, source, training_fraction=0.9, validation_fraction=0.05):
        cls.validate(source, training_fraction, validation_fraction)
        m = len(source)

        num_train = int(round(training_fraction * m))
        num_val = int(round(validation_fraction * m))
        num_test = m - num_train - num_val

        counts = [num_train, num_val, num_test]

        max_count = max(counts)
        min_count = min(counts)

        while min_count == 0:
            max_index = max([i for i, c in enumerate(counts) if c == max_count])
            min_index = max([i for i, c in enumerate(counts) if c == min_count])
            counts[min_index] += 1
            counts[max_index] -= 1

            max_count = max(counts)
            min_count = min(counts)

        num_train, num_val, num_test = counts
        return cls(source, num_train, num_val)

    def __init__(self, data_iterator, num_train, num_val):
        self._iter = data_iterator

        num_test = len(data_iterator) - (num_train + num_val)

        self._counts = [num_train, num_val, num_test]

        buffer_factory = self.get_buffer_factory()

        self._train = buffer_factory.get_train_buffer()
        self._val = buffer_factory.get_validation_buffer()
        self._test = buffer_factory.get_test_buffer()

    def get_buffer_factory(self):
        root = os.path.join('./temp', 'split')
        return H5pyBufferFactory(root)

    def split(self):
        destination = (self._train, self._val, self._test)

        gen = self._iter.get_sequences()

        for buffer_index in range(len(self._counts)):
            while self._counts[buffer_index] > 0:
                points, transcription = next(gen)
                data_set = destination[buffer_index]
                data_set.add_example(points, transcription)
                self._counts[buffer_index] -= 1

        if len(self._val) == 0:
            self._val.add_example(*self._train.pop())

        if len(self._test) == 0:
            self._test.add_example(*self._train.pop())

        assert len(self._train) > 0

    def _create_iterator(self, data_set):
        return H5pySource(data_set, random_order=False)

    def train_data(self):
        return self._create_iterator(self._train)

    def validation_data(self):
        return self._create_iterator(self._val)

    def test_data(self):
        return self._create_iterator(self._test)


class BadFractionsException(Exception):
    pass


class InsufficientNumberOfExamplesException(Exception):
    pass


class BaseFactory:
    def __init__(self, data_source, char_table, num_examples=2000, save_images_path=None):
        self._data_source = data_source

        self._char_table = char_table
        self._num_examples = num_examples
        self._save_images_path = save_images_path

        self._preprocessor = None
        self._train_iter = None
        self._val_iter = None
        self._test_iter = None

        self._Tx = 0
        self._Ty = 0

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
        splitter = DataSplitter(it)

        train_iter = splitter.train_data()

        self._preprocessor = self._get_preprocessor()

        hand_writings = []
        transcriptions = []
        for hwr, t in train_iter.get_sequences():
            hand_writings.append(hwr)
            transcriptions.append(t)

        self._preprocessor.fit(hand_writings, transcriptions)
        hand_writings, transcriptions = self._preprocessor.process(hand_writings, transcriptions)

        self._train_iter = train_iter
        self._val_iter = splitter.validation_data()
        self._test_iter = splitter.test_data()

        self._Tx = self._max_input_len(hand_writings)
        self._Ty = self._max_output_len(transcriptions)

        self._Ty += 1

    def _max_input_len(self, hand_writings):
        return max([len(inp) for inp in hand_writings])

    def _max_output_len(self, transcriptions):
        return max([len(t) for t in transcriptions])

    def _get_preprocessor(self):
        conf = Config()
        config_dict = conf.config_dict

        preprocessor = PreProcessor()

        for entry in config_dict['preprocessors']:
            step_class_name = entry['name']
            cls = getattr(preprocessing, step_class_name)
            params = entry['params']
            preprocessor.add_step(cls(**params))

        return preprocessor

    def save_points(self, points, transcription):
        file_name = '{}.jpg'.format(quote(transcription, safe=' ,.'))
        path = os.path.join(self._save_images_path, file_name)

        points_to_image(points).save(path)

    def _preload(self):
        hand_writings = []
        transcriptions = []
        for i, (points, t) in enumerate(self._data_source.get_sequences()):
            if i > self._num_examples:
                break

            hand_writings.append(points)
            transcriptions.append(t)

        return PreLoadedSource(hand_writings, transcriptions)


class Seq2seqFactory(BaseFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_generator(self):
        return DataSetGenerator(self._train_iter, self._char_table, self._preprocessor, channels=2)

    def validation_generator(self):
        return DataSetGenerator(self._val_iter,
                                self._char_table, self._preprocessor, channels=2)

    def test_generator(self):
        return DataSetGenerator(self._test_iter,
                                self._char_table, self._preprocessor, channels=2)

    def create_model(self):
        from models.seq2seq import SequenceToSequenceTrainer
        return SequenceToSequenceTrainer(self._char_table, input_channels=2)


class AttentionalSeq2seqFactory(BaseFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        conf = Config()
        self._num_cells = conf.config_dict['attention_model']['cells']

    def create_model(self):
        return Seq2SeqWithAttention(self._char_table, self._num_cells,
                                    Tx=self._Tx, Ty=self._Ty, channels=2)

    def _get_generator(self, iterator):
        return AttentionModelDataGenerator(iterator, self._char_table, self._preprocessor,
                                           self._Tx, self._Ty, self._num_cells, channels=2)

    def training_generator(self):
        return self._get_generator(self._train_iter)

    def validation_generator(self):
        return self._get_generator(self._val_iter)

    def test_generator(self):
        return self._get_generator(self._test_iter)