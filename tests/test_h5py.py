import unittest
import os
from data.factories import H5pyRank3DataSet


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
