from unittest import TestCase
from data.example_adapters import ExampleAdapter
from data.generators import MiniBatchGenerator
from sources.preloaded import PreLoadedSource


class AdapterMock(ExampleAdapter):
    def adapt_batch(self, seqs_in, seqs_out):
        return seqs_in, seqs_out


class MiniBatchGeneratorTests(TestCase):
    def test_generator_len(self):
        source = PreLoadedSource([0] * 10, [0] * 10)

        adapter = AdapterMock()
        batch_generator = MiniBatchGenerator(source, adapter)

        self.assertEqual(len(batch_generator), len(source))

    def test_size_of_mini_batches(self):
        num_examples = 10

        source = PreLoadedSource([0] * num_examples, [0] * num_examples)
        adapter = AdapterMock()
        batch_size = 4
        batch_generator = MiniBatchGenerator(source, adapter,
                                             batch_size=batch_size)

        batches = []
        for mini_batch in batch_generator.get_examples():
            if len(batches) >= len(batch_generator) // batch_size + 1:
                break
            batches.append(mini_batch)

        self.assertEqual(len(batches[0][0]), batch_size)
        self.assertEqual(len(batches[1][0]), batch_size)
        self.assertEqual(len(batches[2][0]), num_examples % batch_size)

        self.assertEqual(len(batches[0][1]), batch_size)
        self.assertEqual(len(batches[1][1]), batch_size)
        self.assertEqual(len(batches[2][1]), num_examples % batch_size)

    def test_values_in_mini_batches(self):
        num_examples = 1
        source = PreLoadedSource([0] * num_examples, [0] * num_examples)

        adapter = AdapterMock()
        batch_size = 2
        batch_generator = MiniBatchGenerator(source, adapter,
                                             batch_size=batch_size)

        batches = []
        for mini_batch in batch_generator.get_examples():
            if len(batches) >= len(batch_generator) // batch_size + 1:
                break
            batches.append(mini_batch)

        self.assertEqual(batches[0][0], [0])

        self.assertEqual(batches[0][1], [0])
