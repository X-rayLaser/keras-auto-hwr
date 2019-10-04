from unittest import TestCase
import numpy as np
from keras.utils import to_categorical
from data.char_table import CharacterTable
from data.example_adapters import EmptyListException, BadInputShapeException, CTCAdapter


class CTCAdapterTests(TestCase):
    def setUp(self):
        self.seqs_in = [
            [[0, 1], [1, 2]],
            [[2, 4]]
        ]

        self.seqs_out = ['A', 'bc']

        self.expected_X = np.array([
            self.seqs_in[0],
            self.seqs_in[1] + [[0, 0]]
        ])

        char_table = CharacterTable()

        expected_labels = [
            [char_table.encode('A'), char_table.encode(char_table.sentinel)],
            [char_table.encode('b'), char_table.encode('c')]
        ]

        x_len1 = len(self.seqs_in[0])
        x_len2 = len(self.seqs_in[1])

        y_len1 = len(self.seqs_out[0])
        y_len2 = len(self.seqs_out[1])

        max_y_len = max(y_len1, y_len2)

        self.expected_labels = to_categorical(
            expected_labels, len(char_table)
        ).reshape(max_y_len, max_y_len, len(char_table))

        self.expected_input_lengths = np.array([[x_len1], [x_len2]])
        self.expected_label_lengths = np.array([[y_len1], [y_len2]])

    def test_adapt_input(self):
        adapter = CTCAdapter()

        xs = [[0, 1, 3], [1, 2, 4]]
        res = adapter.adapt_input(xs)
        expected_shape = np.array([xs]).shape
        expected_res = np.array(xs).reshape(expected_shape)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape, expected_shape)
        self.assertEqual(res.tolist(), expected_res.tolist())

    def test_adapt_input_with_invalid_shape(self):
        adapter = CTCAdapter()

        xs = [[[0, 1, 3], [1, 2, 4]]]

        self.assertRaises(EmptyListException,
                          lambda: adapter.adapt_input([]))

        self.assertRaises(BadInputShapeException,
                          lambda: adapter.adapt_input(xs))

        self.assertRaises(BadInputShapeException,
                          lambda: adapter.adapt_input([[]]))

    def test_adapt_batch_called_with_invalid_parameters(self):
        adapter = CTCAdapter()

        self.assertRaises(EmptyListException,
                          lambda: adapter.adapt_batch([], ''))

        self.assertRaises(BadInputShapeException,
                          lambda: adapter.adapt_batch([3], ''))

        self.assertRaises(BadInputShapeException,
                          lambda: adapter.adapt_batch([[[3, 4], [2]]], ''))

        self.assertRaises(BadInputShapeException,
                          lambda: adapter.adapt_batch([[[3, 4], [2]]], ''))

        self.assertRaises(BadInputShapeException,
                          lambda: adapter.adapt_batch([[[3, 4], []]], ''))

    def test_adapt_batch_returns_correct_X(self):
        adapter = CTCAdapter()

        res_inp, _ = adapter.adapt_batch(self.seqs_in, self.seqs_out)

        X = res_inp[0]

        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.shape, self.expected_X.shape)
        self.assertEqual(X.tolist(), self.expected_X.tolist())

    def test_adapt_batch_does_not_modify_passed_arguments(self):
        adapter = CTCAdapter()

        before = [list(s) for s in self.seqs_in]
        adapter.adapt_batch(self.seqs_in, self.seqs_out)
        after = self.seqs_in

        self.assertEqual(before, after)

    def test_adapt_batch_returns_correct_labels(self):
        adapter = CTCAdapter()
        res_inp, _ = adapter.adapt_batch(self.seqs_in, self.seqs_out)

        labels = res_inp[1]

        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(labels.tolist(), self.expected_labels.tolist())

    def test_adapt_batch_returns_correct_input_lengths(self):
        adapter = CTCAdapter()
        res_inp, _ = adapter.adapt_batch(self.seqs_in, self.seqs_out)

        input_lengths = res_inp[2]

        self.assertIsInstance(input_lengths, np.ndarray)
        self.assertEqual(input_lengths.shape,
                         self.expected_input_lengths.shape)
        self.assertEqual(input_lengths.tolist(),
                         self.expected_input_lengths.tolist())

    def test_adapt_batch_returns_correct_label_lengths(self):
        adapter = CTCAdapter()
        res_inp, _ = adapter.adapt_batch(self.seqs_in, self.seqs_out)

        label_lengths = res_inp[3]

        self.assertIsInstance(label_lengths, np.ndarray)
        self.assertEqual(label_lengths.tolist(),
                         self.expected_label_lengths.tolist())


# todo: finish test that tests adapter when input is shorter than 2 * input_length + 1
# todo: cleanup ugly nested lists
