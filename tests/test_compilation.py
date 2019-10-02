from unittest import TestCase
from sources.compiled import CompilationSource, H5pyDataSet
import os
from sources.base import BaseSource
from data.preprocessing import PreProcessor


class MockSource(BaseSource):
    def __init__(self, examples):
        self.examples = examples

    def get_sequences(self):
        for x, y in self.examples:
            yield x, y


class CompilationTests(TestCase):
    def setUp(self):
        self.destination = './test_ds.h5py'

        examples = [
            ([[1], [2], [3]], '123'),
            ([[2]], '2')
        ]
        source = MockSource(examples)
        CompilationSource.compile_data(source, self.destination)

    def tearDown(self):
        if os.path.isfile(self.destination):
            os.remove(self.destination)

    def test_compiles_and_returns_expected_sequences(self):
        compilation = CompilationSource(self.destination, 1, random_order=False)

        examples = list(compilation.get_sequences())
        x, y = examples[0]

        self.assertEqual(x[0], [1])
        self.assertEqual(x[1], [2])
        self.assertEqual(x[2], [3])
        self.assertEqual(y, '123')

    def test_len(self):
        compilation = CompilationSource(self.destination, 1, random_order=False)

        self.assertEqual(len(compilation), 1)

        compilation = CompilationSource(self.destination, 2, random_order=False)

        self.assertEqual(len(compilation), 2)

        compilation = CompilationSource(self.destination, 22, random_order=False)
        self.assertEqual(len(compilation), 2)

    def test_shape(self):
        compilation = CompilationSource(self.destination, 2, random_order=False)
        self.assertEqual(compilation.batch_shape, (2, None, 1))

        source = MockSource([
            ([[2, 4]], '2')
        ])
        CompilationSource.compile_data(source, self.destination)

        compilation = CompilationSource(self.destination, 2, random_order=False)
        self.assertEqual(compilation.batch_shape, (1, None, 2))


# todo: rethink the API of CompilationSource
