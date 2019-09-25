from sources.iam_online import OnlineSource, LinesSource
from sources.wrappers import Normalizer, OffsetPointsSource
from sources.base import BaseSourceWrapper
from sources.compiled import CompilationSource
import os
from data.factories import DataSplitter


class ConstrainedSource(BaseSourceWrapper):
    def __init__(self, source, num_lines):
        super().__init__(source)
        self._num_lines = num_lines

    def get_sequences(self):
        for j, (seq_in, seq_out) in enumerate(self._source.get_sequences()):
            if j % 500 == 0:
                print('Fetched {} examples'.format(j))
            if j >= self._num_lines:
                break
            yield seq_in, seq_out


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./')
    parser.add_argument('--destination_dir', type=str, default='./')
    parser.add_argument('--num_lines', type=int, default=8)
    args = parser.parse_args()

    it = LinesSource(OnlineSource(args.data_path))
    root_source = ConstrainedSource(source=it, num_lines=args.num_lines)

    offset_source = OffsetPointsSource(root_source)

    dest_root = args.destination_dir
    file_names = ['train.h5py', 'validation.h5py', 'test.h5py']
    destinations = [os.path.join(dest_root, f) for f in file_names]

    normalizer = Normalizer()
    splitter = DataSplitter(offset_source)

    sources = [splitter.train_data(), splitter.validation_data(),
               splitter.test_data()]

    train_source, _, _ = sources

    xs = [in_seq for in_seq, _ in train_source.get_sequences()]
    normalizer.fit(xs)
    mu_sd_destination = os.path.join(dest_root, 'mu_std.json')
    normalizer.to_json(mu_sd_destination)

    for i in range(len(file_names)):
        CompilationSource.compile_data(sources[i], destinations[i], normalizer)
