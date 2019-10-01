from sources import BaseSource
import h5py
import random
import numpy as np
import os


class H5pyDataSet:
    def __init__(self, path):
        self._path = path

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
            string_dt = h5py.special_dtype(vlen=str)
            x_dset = x_rows.create_dataset(str(m), data=np.array(xs))
            y_dset = y_rows.create_dataset(str(m), shape=(1,), dtype=string_dt)
            y_dset[0] = ys
            x_dset.flush()
            y_dset.flush()

    def get_example(self, index):
        with h5py.File(self._path, 'r') as f:
            x_rows = f['X_rows']
            y_rows = f['Y_rows']

            dict_key = str(index)
            x = x_rows[dict_key]
            y = y_rows[dict_key]
            return x[:], y[0]

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


class CompilationSource(BaseSource):
    def __init__(self, path, num_lines, random_order=True):
        self._path = path
        self._num_lines = num_lines
        self._size = None
        self._random = random_order

    @staticmethod
    def compile_data(source, destination):
        h5py_data = H5pyDataSet.create(destination)

        for xs, ys in source.get_sequences():
            if len(h5py_data) % 500 == 0:
                print('Compiled {} examples'.format(len(h5py_data)))

            h5py_data.add_example(xs, ys)

    def get_sequences(self):
        h5py_data = H5pyDataSet(self._path)
        return h5py_data.get_data(self._num_lines, self._random)

    def __len__(self):
        h5py_data = H5pyDataSet(self._path)
        return min(self._num_lines, len(h5py_data))
