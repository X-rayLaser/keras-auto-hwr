from unittest import TestCase
from data.char_table import CharacterTable
from models.ctc_model import CTCOutputDecoder


class CTCOutputDecoderTests(TestCase):
    def setUp(self):
        self.original_text = "Hello, world!"

    def test_decode_labels(self):
        char_table = CharacterTable()

        decoder = CTCOutputDecoder(char_table)

        blank = len(char_table)
        num_repeated = 4
        labels = [blank] * num_repeated
        for ch in self.original_text:
            label = char_table.encode(ch)
            labels.extend([label] * num_repeated)
            labels.extend([blank] * num_repeated)

        self.assertEqual(decoder.decode(labels), self.original_text)

    def test_decode_one_label_sequence(self):
        char_table = CharacterTable()

        decoder = CTCOutputDecoder(char_table)

        original = 'c'
        labels = [char_table.encode(original)]

        self.assertEqual(decoder.decode(labels), original)

    def test_decode_sequence_of_blanks(self):
        char_table = CharacterTable()

        decoder = CTCOutputDecoder(char_table)

        blanks = [len(char_table)] * 50

        self.assertEqual(decoder.decode(blanks), '')

    def test_decode_empty_sequence(self):
        char_table = CharacterTable()
        decoder = CTCOutputDecoder(char_table)
        labels = []
        self.assertEqual(decoder.decode(labels), '')

    def test_decode_without_blanks_and_repeatitions(self):
        char_table = CharacterTable()

        decoder = CTCOutputDecoder(char_table)

        labels = [char_table.encode(ch) for ch in self.original_text]

        self.assertEqual(decoder.decode(labels), 'Helo, world!')

    def test_decode_sequence_with_unknown_characters(self):
        char_table = CharacterTable()
        decoder = CTCOutputDecoder(char_table)
        labels = [165, 123568, 123586]
        self.assertEqual(decoder.decode(labels), '???')
