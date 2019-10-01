from data.char_table import CharacterTable
import numpy as np
import unittest
from sources.wrappers import Normalizer
import os
from data.preprocessing import PreProcessor, ProcessingStep,\
    OffsetStep, NormalizationStep, DummyStep
from sources.preloaded import PreLoadedSource
import json
from data.factories import H5pyRank3DataSet


class CharacterTableTests(unittest.TestCase):
    def test_restoration_integrity(self):
        char_table = CharacterTable()

        char_original = 'c'
        char_restored = char_table.decode(char_table.encode(char_original))

        self.assertEqual(char_restored, char_original)

    def test_consistency(self):
        char_original = 'a'

        first_encoding = CharacterTable().encode(char_original)
        second_encoding = CharacterTable().encode(char_original)

        self.assertEqual(first_encoding, second_encoding)

    def test_decode_out_of_alphabet(self):
        char_table = CharacterTable()
        res = char_table.decode(len(char_table))
        self.assertEqual(res, '?')

    def test_mapping_is_one_to_one(self):
        char_table = CharacterTable()
        decoded_chars = []
        for code in range(len(char_table)):
            ch = char_table.decode(code)
            decoded_chars.append(ch)

        self.assertEqual(
            len(decoded_chars), len(set(decoded_chars)),
            'Got duplicate characters from different codes: {}'.format(
                decoded_chars)
        )

        encoded_chars = []
        for ch in decoded_chars:
            encoded_chars.append(char_table.encode(ch))

        self.assertEqual(
            len(encoded_chars), len(set(encoded_chars)),
            '2 or more characters got mapped to the same code:'.format(
                encoded_chars)
        )

    def test_sentinel(self):
        char_table = CharacterTable()
        sentinel = char_table.sentinel

        decoded = char_table.decode(char_table.encode(sentinel))
        self.assertEqual(decoded, sentinel)


class NormalizerTests(unittest.TestCase):
    def test_fit(self):
        normalizer = Normalizer()

        dummy = [[1, 4], [1, 3], [1, 5]]
        X = np.array([dummy])
        normalizer.fit(X)

        self.assertEqual(normalizer.mu.tolist(), [1, 4])
        self.assertAlmostEqual(normalizer.sd.tolist(), [0, 0.816496580927726])

    def test_normalize(self):
        normalizer = Normalizer()

        normalizer.set_mean([1, 2])
        normalizer.set_deviation([4, 1])

        dummy = [[1, 2], [2, 10]]
        X = np.array([dummy])
        res = normalizer.preprocess(X)

        self.assertEqual(res[0][0], [0, 2])
        self.assertEqual(res[0][1], [0.25, 10])

    def test_serrialization(self):
        normalizer = Normalizer()
        normalizer.set_mean([1, 2])
        normalizer.set_deviation([4, 1])

        path = './test_mu.json'
        normalizer.to_json(path)

        normalizer = Normalizer.from_json(path)

        self.assertIsInstance(normalizer.mu, np.ndarray)
        self.assertIsInstance(normalizer.sd, np.ndarray)

        self.assertEqual(normalizer.mu.tolist(), [1, 2])
        self.assertEqual(normalizer.sd.tolist(), [4, 1])

        if os.path.isfile(path):
            os.remove(path)


class PreProcessorTests(unittest.TestCase):
    class AddingStep(ProcessingStep):
        def fit(self, data):
            pass

        def process_example(self, x, y):
            return x + 1, y + 1

    class MultiplyingStep(ProcessingStep):
        def fit(self, data):
            pass

        def process_example(self, x, y):
            return x * 2, y * 2

    class DummyStorage:
        def save(self, s):
            self.data = s

        def load(self):
            return self.data

    def test_process_example_with_one_step(self):
        preprocessor = PreProcessor(steps=[self.AddingStep()])
        x, y = preprocessor.pre_process_example(2, 5)
        self.assertEqual(x, 3)
        self.assertEqual(y, 6)

    def test_process_example_with_2_steps(self):
        preprocessor = PreProcessor(steps=[self.AddingStep(),
                                           self.MultiplyingStep()])
        x, y = preprocessor.pre_process_example(2, 5)
        self.assertEqual(x, 6)
        self.assertEqual(y, 12)

    def test_order_of_steps_matters(self):
        preprocessor = PreProcessor(steps=[self.MultiplyingStep(),
                                           self.AddingStep()])
        x, y = preprocessor.pre_process_example(2, 5)
        self.assertEqual(x, 5)
        self.assertEqual(y, 11)

    def test_fit_and_process_example(self):
        preprocessor = PreProcessor(steps=[DummyStep(), DummyStep()])

        xs = [1, 2]
        ys = [0, 0]
        source = PreLoadedSource(xs, ys)
        preprocessor.fit(source)
        x, y = preprocessor.pre_process_example(1, 2)

        self.assertEqual(x, 13)
        self.assertEqual(y, 2)

    def test_save(self):
        step1 = DummyStep()
        step1.s = 3
        step2 = DummyStep()
        step2.s = 9

        preprocessor = PreProcessor(steps=[step1, step2])

        storage = self.DummyStorage()

        preprocessor.save(storage)

        expected = {
            'steps': [{
                    'class_name': 'DummyStep',
                    'params': {'sum': 3}
            }, {
                'class_name': 'DummyStep',
                'params': {'sum': 9}
            }]
        }

        self.assertEqual(storage.data, json.dumps(expected))

    def test_load(self):
        preprocessor = PreProcessor(steps=[])

        storage = self.DummyStorage()

        expected = {
            'steps': [{
                'class_name': 'DummyStep',
                'params': {'sum': 3}
            }, {
                'class_name': 'DummyStep',
                'params': {'sum': 9}
            }]
        }
        storage.data = json.dumps(expected)

        preprocessor.load(storage)

        self.assertEqual(len(preprocessor.steps), 2)
        self.assertIsInstance(preprocessor.steps[0], DummyStep)
        self.assertIsInstance(preprocessor.steps[1], DummyStep)

        self.assertEqual(preprocessor.steps[0].s, 3)
        self.assertEqual(preprocessor.steps[1].s, 9)


class H5pyRank3DataSetTests(unittest.TestCase):
    def setUp(self):
        self.temp = './temp.h5py'
        self.ds = H5pyRank3DataSet.create(self.temp)

    def tearDown(self):
        if os.path.isfile(self.temp):
            os.remove(self.temp)

    def test_persistence(self):
        ds = self.ds

        strokes = [
            [[2, 5], [3, 9]], [[1, 2]]
        ]
        text = 'hello'
        ds.add_example(strokes, text)

        strokes_restored, text_restored = ds.get_example(0)

        self.assertEqual(strokes, strokes_restored)
        self.assertEqual(text, text_restored)

    def test_get_all_examples(self):
        ds = self.ds

        first_stroke_line = [
            [[2, 5], [3, 9]], [[1, 2]]
        ]

        second_stroke_line = [
            [[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3], [1, 2, 3]]
        ]

        first_text = 'one'
        second_text = 'two'

        ds.add_example(first_stroke_line, first_text)
        ds.add_example(second_stroke_line, second_text)

        gen = ds.get_data(random_order=False)

        restored_strokes = []
        restored_texts = []
        for xs, ys in gen:
            restored_strokes.append(xs)
            restored_texts.append(ys)

        self.assertEqual(restored_strokes[0], first_stroke_line)
        self.assertEqual(restored_texts[0], first_text)
        self.assertEqual(restored_strokes[1], second_stroke_line)
        self.assertEqual(restored_texts[1], second_text)


class CompilationSource(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
