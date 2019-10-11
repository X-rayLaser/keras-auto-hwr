from unittest import TestCase

from data.compiler import DataSetCompiler
from sources.preloaded import PreLoadedSource
from data.preprocessing import PreProcessor


class DataRepoMock:
    def __init__(self):
        self.slices = []

    def add_slice(self):
        self.slices.append([])

    def add_example(self, slice_index, x, y):
        self.slices[slice_index].append((x, y))


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
        compiler = DataSetCompiler(preprocessor, splitter, repo)

        compiler.compile()
        self.assertTrue(splitter.called)

    def test_preprocessor_fits_training_data(self):
        preprocessor = PreprocessorMock([])
        provider = PreLoadedSource([2, 4], ["two four"])
        repo = DataRepoMock()
        splitter = DataSplitterMock(provider)

        compiler = DataSetCompiler(preprocessor, splitter, repo)
        compiler.compile()
        self.assertEqual(list(preprocessor.fit_arg.get_sequences()), [(1, '1')])

    def test_data_processing_calls(self):
        preprocessor = PreprocessorMock([])
        provider = PreLoadedSource([2, 4], ["two four"])
        repo = DataRepoMock()
        splitter = DataSplitterMock(provider)

        compiler = DataSetCompiler(preprocessor, splitter, repo)
        compiler.compile()
        self.assertEqual(preprocessor.processed_args, [(1, '1'), (2, '2'), (3, '3')])

    def test_repo(self):
        preprocessor = PreProcessor([])
        provider = PreLoadedSource([1, 2, 3], ['1', '2', '3'])
        repo = DataRepoMock()
        splitter = DataSplitterMock(provider)

        compiler = DataSetCompiler(preprocessor, splitter, repo)

        compiler.compile()

        self.assertEqual(len(repo.slices), 3)

        train, val, test = repo.slices

        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 1)
        self.assertEqual(len(test), 1)

        self.assertEqual(train, [(1, '1')])
        self.assertEqual(val, [(2, '2')])
        self.assertEqual(test, [(3, '3')])


# todo: decorator class for provider that performs random shuffling using HDD as buffer
# todo: use this decorator to pass it to splitter
