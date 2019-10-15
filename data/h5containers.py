import os
import random

import h5py
import numpy as np


class DataRepo:
    def add_example(self, xs, ys):
        raise NotImplementedError

    def get_example(self, index):
        raise NotImplementedError

    def get_data(self, num_lines=None, random_order=False):
        raise NotImplementedError


class H5pyDataSet:
    def __init__(self, path):
        self._path = path

    @property
    def location(self):
        return self._path

    @staticmethod
    def create(path):
        if os.path.isfile(path):
            os.remove(path)

        with h5py.File(path, 'w') as f:
            f.create_group('X_rows')
            f.create_group('Y_rows')

        return H5pyDataSet(path)

    def __len__(self):
        with h5py.File(self._path, 'r') as f:
            y_rows = f['Y_rows']
            return len(y_rows.keys())

    def add_example(self, xs, ys):
        with h5py.File(self._path, 'a') as f:
            x_rows = f['X_rows']
            y_rows = f['Y_rows']

            m = len(self)

            x_dset = x_rows.create_dataset(str(m), data=np.array(xs))

            if type(ys) == str:
                dtype = h5py.special_dtype(vlen=str)
                y_dset = y_rows.create_dataset(str(m), shape=(1,), dtype=dtype)
                y_dset[0] = ys
            else:
                y_dset = y_rows.create_dataset(str(m), data=np.array(ys))

            x_dset.flush()
            y_dset.flush()

    def get_example(self, index):
        with h5py.File(self._path, 'r') as f:
            x_rows = f['X_rows']
            y_rows = f['Y_rows']

            dict_key = str(index)
            x = x_rows[dict_key]
            y = y_rows[dict_key]

            # todo: refactor this and similar one in a method above
            if type(y[0]) is str:
                return x[:], y[0]
            else:
                return x[:], y[:]

    def get_data(self, num_lines=None, random_order=False):
        if num_lines is None:
            m = len(self)
        else:
            m = min(num_lines, len(self))

        indices = list(range(m))

        if random_order:
            random.shuffle(indices)

        for index in indices:
            yield self.get_example(index)

    def pop(self):
        with h5py.File(self._path, 'a') as f:
            x_rows = f['X_rows']
            y_rows = f['Y_rows']

            last_index = len(self) - 1
            dict_key = str(last_index)
            xs = x_rows[dict_key]
            ys = y_rows[dict_key]

            del x_rows[dict_key]
            del y_rows[dict_key]
            return xs, ys


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