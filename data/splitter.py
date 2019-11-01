import os
from data.h5containers import H5pyRank3DataSet
from sources.wrappers import H5pySource


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
        root = os.path.join(os.getcwd(), 'temp', 'split')
        if not os.path.exists(root):
            os.makedirs(root)
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
