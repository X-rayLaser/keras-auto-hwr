from unittest import TestCase
from sources.preloaded import PreLoadedSource
from data.preprocessing import PreProcessor
from sources.base import BaseSourceWrapper


class DataRepoMock:
    def __init__(self):
        self.slices = []

    def add_slice(self):
        self.slices.append([])

    def add_example(self, slice_index, x, y):
        self.slices[slice_index].append((x, y))


class PreprocessedSource(BaseSourceWrapper):
    def __init__(self, source, preprocessor):
        super().__init__(source)
        self._preprocessor = preprocessor

    def get_sequences(self):
        for xs, ys in self._source.get_sequences():
            yield self._preprocessor.pre_process_example(xs, ys)


class DataSetCompiler:
    def __init__(self, provider, preprocessor, splitter, repo):
        self._provider = provider
        self._preprocessor = preprocessor
        self._splitter = splitter
        self._repo = repo

    def compile(self):
        self._splitter.split()
        self._preprocessor.fit(self._splitter.train_data())

        train_source = PreprocessedSource(self._splitter.train_data(),
                                          self._preprocessor)
        val_source = PreprocessedSource(self._splitter.validation_data(),
                                        self._preprocessor)

        test_source = PreprocessedSource(self._splitter.test_data(),
                                         self._preprocessor)

        self._repo.add_slice()
        self._repo.add_slice()
        self._repo.add_slice()

        for xs, ys in train_source.get_sequences():
            self._repo.add_example(0, xs, ys)

        for xs, ys in val_source.get_sequences():
            self._repo.add_example(1, xs, ys)

        for xs, ys in test_source.get_sequences():
            self._repo.add_example(2, xs, ys)


class PreprocessorMock(PreProcessor):
    def __init__(self, steps):
        super().__init__(steps)

        self.processed_args = []
        self.fit_arg = None

    def fit(self, source):
        self.fit_arg = source

    def pre_process_example(self, x, y):
        self.processed_args.append((x, y))
        return x, y


class DataSplitterMock:
    def __init__(self, provider):
        self.called = False
        self._provider = provider
        self._slices = []

    def split(self):
        self.called = True
        self._slices = list(self._provider.get_sequences())

    def train_data(self):
        return PreLoadedSource([1], ['1'])

    def validation_data(self):
        return PreLoadedSource([2], ['2'])

    def test_data(self):
        return PreLoadedSource([3], ['3'])


class DataSetCompilerTests(TestCase):
    def test_compiler_performs_splits_data(self):
        preprocessor = PreprocessorMock([])
        x = [1, 2, 3]
        y = ['one', 'two', 'three']
        provider = PreLoadedSource(x, y)
        repo = DataRepoMock()

        splitter = DataSplitterMock(provider)
        compiler = DataSetCompiler(provider, preprocessor, splitter, repo)

        compiler.compile()
        self.assertTrue(splitter.called)

    def test_preprocessor_fits_training_data(self):
        preprocessor = PreprocessorMock([])
        provider = PreLoadedSource([2, 4], ["two four"])
        repo = DataRepoMock()
        splitter = DataSplitterMock(provider)

        compiler = DataSetCompiler(provider, preprocessor, splitter, repo)
        compiler.compile()
        self.assertEqual(list(preprocessor.fit_arg.get_sequences()), [(1, '1')])

    def test_data_processing_calls(self):
        preprocessor = PreprocessorMock([])
        provider = PreLoadedSource([2, 4], ["two four"])
        repo = DataRepoMock()
        splitter = DataSplitterMock(provider)

        compiler = DataSetCompiler(provider, preprocessor, splitter, repo)
        compiler.compile()
        self.assertEqual(preprocessor.processed_args, [(1, '1'), (2, '2'), (3, '3')])

    def test_repo(self):
        preprocessor = PreProcessor([])
        provider = PreLoadedSource([1, 2, 3], ['1', '2', '3'])
        repo = DataRepoMock()
        splitter = DataSplitterMock(provider)

        compiler = DataSetCompiler(provider, preprocessor, splitter, repo)

        compiler.compile()

        self.assertEqual(len(repo.slices), 3)

        train, val, test = repo.slices

        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 1)
        self.assertEqual(len(test), 1)

        self.assertEqual(train, [(1, '1')])
        self.assertEqual(val, [(2, '2')])
        self.assertEqual(test, [(3, '3')])
