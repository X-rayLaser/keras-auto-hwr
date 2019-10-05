from unittest import TestCase
from data.char_table import CharacterTable


class CTCOutputDecoder:
    def __init__(self, char_table):
        self._char_table = char_table

    def remove_repeated(self, labels):
        prev = -1
        res = []
        for label in labels:
            if label != prev:
                res.append(label)
                prev = label
        return res

    def remove_blanks(self, labels):
        return [label for label in labels if label != len(self._char_table)]

    def decode(self, labels):
        labels = self.remove_repeated(labels)
        labels = self.remove_blanks(labels)

        characters = [self._char_table.decode(label) for label in labels]
        return ''.join(characters)


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
