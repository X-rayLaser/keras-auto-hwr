from data.encodings import CharacterTable
import unittest
from data.encodings import TextEncodingTable


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


class WordEncodingTableTests(unittest.TestCase):
    def setUp(self):
        word2code = {
            'apple': 0,
            'orange': 1
        }
        self.encoding_table = TextEncodingTable(word2code)

    def test_len(self):
        self.assertEqual(len(self.encoding_table), 5)

    def test_encode_words_present_in_the_table(self):
        self.assertEqual(self.encoding_table.encode('apple'), 0)
        self.assertEqual(self.encoding_table.encode('orange'), 1)

    def test_encode_word_absent_in_the_table(self):
        self.assertEqual(self.encoding_table.encode('winter'),
                         self.encoding_table.code_for_unknown)

    def test_special_encodings(self):
        self.assertEqual(self.encoding_table.start, 3)
        self.assertEqual(self.encoding_table.sentinel, 4)

    def test_decode_word_codes(self):
        self.assertEqual(self.encoding_table.decode(0), 'apple')
        self.assertEqual(self.encoding_table.decode(1), 'orange')

    def test_decode_unknown(self):
        unknown_code = self.encoding_table.code_for_unknown
        self.assertEqual(self.encoding_table.decode(unknown_code), '**?**')

    def test_decoding_start_of_sequence(self):
        start = self.encoding_table.start
        self.assertEqual(self.encoding_table.decode(start), '**start**')

    def test_decoding_end_of_sequence(self):
        enf_of_seq = self.encoding_table.sentinel
        self.assertEqual(self.encoding_table.decode(enf_of_seq), '**end**')

    def test_decode_sequence(self):
        res = self.encoding_table.decode_sequence([1, 0, 0])
        self.assertEqual(res, ['orange', 'apple', 'apple'])
