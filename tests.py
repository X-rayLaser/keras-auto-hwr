from data.char_table import CharacterTable
import numpy as np
import unittest
from sources.wrappers import Normalizer
import os


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


if __name__ == '__main__':
    unittest.main()
