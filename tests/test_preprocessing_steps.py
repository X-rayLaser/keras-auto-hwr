from unittest import TestCase
from data.preprocessing import WordEncodingStep
from data.preprocessing import UnicodeEncodingStep
from data.preprocessing import NormalizationStep
from data.encodings import TextEncodingTable


class NormalizationStepTests(TestCase):
    def test_set_and_retrieve_parameters(self):
        step = NormalizationStep()
        params = {
            'mu': [0, 0, 0, 1],
            'sd': [1, 2, 1, 1]
        }
        step.set_parameters(params)
        res = step.get_parameters()
        self.assertEqual(params, res)

    def test_preprocess_after_setting_parameters(self):
        step = NormalizationStep()
        params = {
            'mu': [0, 0, 0, 1],
            'sd': [1, 2, 3, 4]
        }
        step.set_parameters(params)
        x = [(1, 2, 3, 1)]
        expected_x = [[1, 1, 1, 1]]
        self.assertEqual(expected_x, step.process_x(x))


class WordEncodingStepTests(TestCase):
    def setUp(self):
        step = WordEncodingStep(num_words=3)

        data = [
            (1, 'Foo bar'),
            (2, 'Foo and and bar'),
            (3, 'Foo A;B.C)')
        ]

        step.fit(data)

        self.step = step
        self.expected_keys = {'Foo', ' ', 'bar'}
        self.expected_values = {0, 1, 2}

    def test_create_word_mapping_table_from_data(self):
        step = self.step

        self.assertEqual(set(step.word2code.keys()), self.expected_keys)
        self.assertEqual(set(step.word2code.values()), self.expected_values)

    def test_process_x_does_nothing(self):
        step = WordEncodingStep(num_words=3)
        x = [5, 6, 7]
        self.assertEqual(step.process_x(x), x)

    def test_process_y_after_fitting(self):
        step = self.step
        labels = step.process_y('Foo bar')
        codes = step.word2code
        self.assertEqual(labels, [codes['Foo'], codes[' '], codes['bar']])

    def test_unknown_words_replaced_by_question_mark(self):
        labels = self.step.process_y('Foo unknown word')
        codes = self.step.word2code
        self.assertEqual(labels, [codes['Foo'], codes[' '],
                                  self.step.unknown_code,
                                  codes[' '],
                                  self.step.unknown_code])

    def test_get_parameters(self):
        word2code = self.step.get_parameters()

        self.assertEqual(set(word2code.keys()), self.expected_keys)
        self.assertEqual(set(word2code.values()), self.expected_values)

    def test_set_parameters(self):
        step = WordEncodingStep(num_words=3)

        d = dict((k, v) for k, v in zip(self.expected_keys, self.expected_values))
        step.set_parameters(d)

        word2code = step.get_parameters()
        self.assertEqual(set(word2code.keys()), self.expected_keys)
        self.assertEqual(set(word2code.values()), self.expected_values)

    def test_fit_using_russian_text(self):
        step = WordEncodingStep(num_words=50, language='russian')

        data = [
            (1, 'Привет, мир'),
            (2, 'Привет;')
        ]

        step.fit(data)
        word2code = step.get_parameters()

        num_tokens = 5
        expected_values = set(range(num_tokens))
        expected_keys = {'Привет', ',', ' ', 'мир', ';'}
        self.assertEqual(set(word2code.keys()), expected_keys)
        self.assertEqual(set(word2code.values()), expected_values)


class CharacterEncodingStepTests(TestCase):
    def setUp(self):
        self.hello_world = 'ハロー・ワールド'
        self.data = [(0, self.hello_world)]
        self.encoding_step = UnicodeEncodingStep()

    def test_process_returns_labels_with_correct_length(self):
        self.encoding_step.fit(self.data)
        res = self.encoding_step.process_y(self.hello_world)
        self.assertEqual(len(self.hello_world), len(res))

    def test_process_japanese_characters(self):
        self.encoding_step.fit(self.data)
        res = self.encoding_step.process_y(self.hello_world)
        expected = [0, 1, 2, 3, 4, 2, 5, 6]
        self.assertEqual(res, expected)

    def test_fit_multiple_examples(self):
        hello_ru = 'Привет мир'
        data = [(0, self.hello_world), (1, hello_ru)]
        self.encoding_step.fit(data)

        first_num = len(set(self.hello_world))
        second_num = len(set(hello_ru))
        expected_size = first_num + second_num

        char2code = self.encoding_step.get_parameters()
        self.assertEqual(len(char2code), expected_size)

    def test_processing_unknown_characters(self):
        self.encoding_step.fit(self.data)

        char2code = self.encoding_step.get_parameters()
        unknown_code = len(char2code)
        expected = [unknown_code] * 3

        res = self.encoding_step.process_y('Xyz')
        self.assertEqual(res, expected)

    def test_before_fitting(self):
        char2code = self.encoding_step.get_parameters()
        self.assertEqual(len(char2code), 0)

    def test_integration_with_encoding_table(self):
        self.encoding_step.fit(self.data)
        params = self.encoding_step.get_parameters()

        encoding_table = TextEncodingTable(params)
        original = self.hello_world

        labels = self.encoding_step.process_y(original)
        reconstruction = ''.join(encoding_table.decode_sequence(labels))

        self.assertEqual(original, reconstruction)


# todo: remove create_source duplication
# todo: implement or integrate beam search for CTC model decoding (use character-level model)
# todo: implement or integrate token passing alrogithm
