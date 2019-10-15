from unittest import TestCase
from data.splitter import DataSplitter, BaseBuffer, BaseBufferFactory
from sources.base import BaseSource
from data.splitter import BadFractionsException
from data.splitter import InsufficientNumberOfExamplesException


class SourceMock(BaseSource):
    def __init__(self):
        self.examples = [(1, '1'), (2, '2'), (3, '3'),
                         (4, '4'), (5, '5'), (6, '6')]

    def get_sequences(self):
        for example in self.examples:
            yield example

    def __len__(self):
        return len(self.examples)


class BufferFactory(BaseBufferFactory):
    def get_buffer(self):
        return InMemoryBuffer()

    def get_train_buffer(self):
        return self.get_buffer()

    def get_validation_buffer(self):
        return self.get_buffer()

    def get_test_buffer(self):
        return self.get_buffer()


class InMemoryBuffer(BaseBuffer):
    def __init__(self):
        self._items = []

    def __len__(self):
        return len(self._items)

    def add_example(self, strokes, transcription_text):
        self._items.append((strokes, transcription_text))

    def pop(self):
        return self._items.pop()

    def get_data(self, num_lines=None, random_order=False):
        for example in self._items:
            yield example


class DataSplitterBuffered(DataSplitter):
    def get_buffer_factory(self):
        return BufferFactory()


class DataSplitterTests(TestCase):
    def setUp(self):
        self.source = SourceMock()
        self.num_train = 3
        self.num_val = 2
        self.num_test = len(self.source) - self.num_train - self.num_val
        self.splitter = DataSplitterBuffered(self.source, self.num_train,
                                             self.num_val)

    def test_returns_slices_of_correct_size(self):
        self.splitter.split()
        self.assertEqual(len(self.splitter.train_data()), self.num_train)
        self.assertEqual(len(self.splitter.validation_data()), self.num_val)
        self.assertEqual(len(self.splitter.test_data()), self.num_test)

    def test_slices_contain_examples_as_expected(self):
        self.splitter.split()

        all_examples = list(self.source.get_sequences())
        expected_train = all_examples[:self.num_train]
        expected_val = all_examples[self.num_train:self.num_train + self.num_val]
        expected_test = all_examples[self.num_train + self.num_val:]

        train_gen = self.splitter.train_data().get_sequences()
        val_gen = self.splitter.validation_data().get_sequences()
        test_gen = self.splitter.test_data().get_sequences()

        self.assertEqual(list(train_gen), expected_train)
        self.assertEqual(list(val_gen), expected_val)
        self.assertEqual(list(test_gen), expected_test)

    def validate_sizes(self, splitter, train_size, val_size, test_size):
        self.assertEqual(len(splitter.train_data()), train_size)
        self.assertEqual(len(splitter.validation_data()), val_size)
        self.assertEqual(len(splitter.test_data()), test_size)

    def test_factory_method(self):
        p_train = 0.5
        p_val = 0.25
        splitter = DataSplitterBuffered.create(
            self.source, training_fraction=p_train, validation_fraction=p_val
        )

        splitter.split()
        total_size = len(self.source)
        train_size = int(round(total_size * p_train))
        val_size = int(round(total_size * p_val))
        test_size = total_size - train_size - val_size

        self.validate_sizes(splitter, train_size, val_size, test_size)

    def test_would_make_zero_train_examples(self):
        splitter = DataSplitterBuffered.create(
            self.source, training_fraction=0.05, validation_fraction=0.9
        )

        splitter.split()
        total_size = len(self.source)
        train_size = 1
        val_size = 4
        test_size = total_size - train_size - val_size

        self.validate_sizes(splitter, train_size, val_size, test_size)

    def test_would_make_zero_train_and_test_examples(self):
        splitter = DataSplitterBuffered.create(
            self.source, training_fraction=0.05, validation_fraction=0.95
        )

        splitter.split()
        total_size = len(self.source)
        train_size = 1
        val_size = 4
        test_size = total_size - train_size - val_size

        self.validate_sizes(splitter, train_size, val_size, test_size)

    def test_would_make_zero_train_and_val_examples(self):
        splitter = DataSplitterBuffered.create(
            self.source, training_fraction=0, validation_fraction=0
        )

        splitter.split()
        total_size = len(self.source)
        train_size = 1
        val_size = 1
        test_size = total_size - train_size - val_size

        self.validate_sizes(splitter, train_size, val_size, test_size)

    def validate_exception(self, source, p_train, p_val, exc):
        self.assertRaises(
            exc,
            lambda: DataSplitterBuffered.create(
                source, training_fraction=p_train, validation_fraction=p_val
            )
        )

    def test_class_method_with_invalid_parameters(self):
        self.validate_exception(self.source, 1.5, 0.3, BadFractionsException)

        self.validate_exception(self.source, 0.2, 1.3, BadFractionsException)

        self.validate_exception(self.source, -0.7, 0.9, BadFractionsException)

        self.validate_exception(self.source, 0.7, -0.9, BadFractionsException)

        self.validate_exception(self.source, 0.7, 0.9, BadFractionsException)

        source = SourceMock()
        source.examples = [(1, '1'), (2, '2')]

        self.validate_exception(
            source, 0.7, 0.1, InsufficientNumberOfExamplesException
        )
