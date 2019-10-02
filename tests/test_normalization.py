import numpy as np
import unittest
from sources.wrappers import Normalizer
import os


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
