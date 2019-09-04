import os
from urllib.parse import quote

import numpy as np

from data import PreProcessor, Flattening, SignalMaker, DeltaSignal, DataSetGenerator, Truncation, SequencePadding, \
    AttentionModelDataGenerator
from models.attention import Seq2SeqWithAttention
from sources.preloaded import PreLoadedSource
from util import points_to_image
from . import preprocessing
from config import Config


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

        from config import Config
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