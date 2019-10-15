from unittest import TestCase
from data.preprocessing import WordEncodingStep


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
        self.expected_keys = {'Foo', 'and', 'bar'}
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
        labels = step.process_y('Foo and bar')
        codes = step.word2code
        self.assertEqual(labels, [codes['Foo'], codes['and'], codes['bar']])

    def test_unknown_words_replaced_by_question_mark(self):
        labels = self.step.process_y('Foo and unknown word')
        codes = self.step.word2code
        self.assertEqual(labels, [codes['Foo'], codes['and'],
                                  self.step.unknown_code,
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


# todo: construct mapping table from word2code dictionary
# todo: use mapping table in output decoder instead of character table
# todo: remove create_source duplication
