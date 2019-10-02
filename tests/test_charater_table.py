from data.char_table import CharacterTable
import unittest


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