from sources import BaseSource
from sources.wrappers import OffsetPointsSource, NormalizedSource
import h5py


class CompilationSource(BaseSource):
    def __init__(self, path, num_lines):
        self._path = path
        self._num_lines = num_lines
        self._size = None

    @staticmethod
    def compile_data(source, destination, normalizer):
        source = NormalizedSource(OffsetPointsSource(source), normalizer)

        with h5py.File(destination, 'w') as f:
            x_rows = f.create_group('X_rows')
            y_rows = f.create_group('Y_rows')

            string_dt = h5py.special_dtype(vlen=str)

            counter = 0
            for xs, transcription in source.get_sequences():
                if counter % 500 == 0:
                    print('Compiled {} examples'.format(counter))

                import numpy as np
                x_dset = x_rows.create_dataset(str(counter), data=np.array(xs))
                y_dset = y_rows.create_dataset(str(counter), shape=(1,), dtype=string_dt)
                y_dset[0] = transcription
                x_dset.flush()
                y_dset.flush()
                counter += 1

    def get_sequences(self):
        with h5py.File(self._path, 'r') as f:
            x_rows = f['X_rows']
            y_rows = f['Y_rows']

            m = min(self._num_lines, len(y_rows.keys()))

            for i in range(m):
                x = x_rows[str(i)]
                y = y_rows[str(i)]
                yield x[:], y[0]

    def __len__(self):
        if not self._size:
            i = 0
            for i, _ in enumerate(self.get_sequences()):
                pass

            self._size = min(self._num_lines, i + 1)
        return self._size
