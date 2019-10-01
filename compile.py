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

        self._use_all = (num_lines == 0)

    def get_sequences(self):
        for j, (seq_in, seq_out) in enumerate(self._source.get_sequences()):
            if j % 500 == 0:
                print('Fetched {} examples'.format(j))

            if j >= self._num_lines and not self._use_all:
                break
            yield seq_in, seq_out


def split_data(source):
    splitter = DataSplitter(source)

    return [splitter.train_data(), splitter.validation_data(),
            splitter.test_data()]


def compile_data(destination_dir, normalizer):
    file_names = ['train.h5py', 'validation.h5py', 'test.h5py']
    destinations = [os.path.join(destination_dir, f) for f in file_names]

    for i in range(len(file_names)):
        source = NormalizedSource(sources[i], normalizer)
        CompilationSource.compile_data(source, destinations[i])


if __name__ == '__main__':
    import argparse
    from sources.wrappers import NormalizedSource

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./')
    parser.add_argument('--destination_dir', type=str, default='./')
    parser.add_argument('--num_lines', type=int, default=8)

    args = parser.parse_args()

    if args.num_lines == 0:
        print('WARNING: num_lines is set to 0, thus all data will be used')

    dest_root = args.destination_dir

    it = LinesSource(OnlineSource(args.data_path))
    root_source = ConstrainedSource(source=it, num_lines=args.num_lines)
    offset_source = OffsetPointsSource(root_source)

    normalizer = Normalizer()

    sources = split_data(offset_source)

    train_source, _, _ = sources

    xs = (in_seq for in_seq, _ in train_source.get_sequences())
    normalizer.fit(xs)
    mu_sd_destination = os.path.join(dest_root, 'mu_std.json')
    normalizer.to_json(mu_sd_destination)

    compile_data(dest_root, normalizer)
