from unittest import TestCase
import numpy as np
from keras.utils import to_categorical
from data.encodings import CharacterTable
from data.example_adapters import EmptyListException, BadInputShapeException, CTCAdapter, Seq2seqAdapter


def make_batch():
    seqs_in = [
        [[0, 1], [1, 2]],
        [[2, 4]]
    ]

    seqs_out = [
        [56],
        [82, 71]
    ]

    return seqs_in, seqs_out


class CTCAdapterTests(TestCase):
    def setUp(self):
        self.seqs_in, self.seqs_out = make_batch()

        self.expected_X = np.array([
            self.seqs_in[0],
            self.seqs_in[1] + [[0, 0]]
        ])

        self.padding_value = 138

        expected_labels = [
            [56, self.padding_value],
            [82, 71]
        ]

        x_len1 = len(self.seqs_in[0])
        x_len2 = len(self.seqs_in[1])

        y_len1 = len(self.seqs_out[0])
        y_len2 = len(self.seqs_out[1])

        max_y_len = max(y_len1, y_len2)

        self.expected_labels = np.array(expected_labels).reshape(max_y_len,
                                                                 max_y_len)

        self.expected_input_lengths = np.array([[x_len1], [x_len2]])
        self.expected_label_lengths = np.array([[y_len1], [y_len2]])
        self.adapter = CTCAdapter(y_padding=self.padding_value)

    def test_adapt_input(self):
        adapter = self.adapter

        xs = [[0, 1, 3], [1, 2, 4]]
        res = adapter.adapt_input(xs)
        expected_shape = np.array([xs]).shape
        expected_res = np.array(xs).reshape(expected_shape)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape, expected_shape)
        self.assertEqual(res.tolist(), expected_res.tolist())

    def test_adapt_input_with_invalid_shape(self):
        adapter = self.adapter

        xs = [[[0, 1, 3], [1, 2, 4]]]

        self.assertRaises(EmptyListException,
                          lambda: adapter.adapt_input([]))

        self.assertRaises(BadInputShapeException,
                          lambda: adapter.adapt_input(xs))

        self.assertRaises(BadInputShapeException,
                          lambda: adapter.adapt_input([[]]))

    def test_adapt_batch_called_with_invalid_parameters(self):
        adapter = self.adapter

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
        adapter = self.adapter

        res_inp, _ = adapter.adapt_batch(self.seqs_in, self.seqs_out)

        X = res_inp[0]

        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.shape, self.expected_X.shape)
        self.assertEqual(X.tolist(), self.expected_X.tolist())

    def test_adapt_batch_does_not_modify_passed_arguments(self):
        adapter = CTCAdapter(self.padding_value)

        before = [list(s) for s in self.seqs_in]
        adapter.adapt_batch(self.seqs_in, self.seqs_out)
        after = self.seqs_in

        self.assertEqual(before, after)

    def test_adapt_batch_returns_correct_labels(self):
        adapter = CTCAdapter(self.padding_value)
        res_inp, _ = adapter.adapt_batch(self.seqs_in, self.seqs_out)

        labels = res_inp[1]

        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(labels.tolist(), self.expected_labels.tolist())

    def test_adapt_batch_returns_correct_input_lengths(self):
        adapter = CTCAdapter(self.padding_value)
        res_inp, _ = adapter.adapt_batch(self.seqs_in, self.seqs_out)

        input_lengths = res_inp[2]

        self.assertIsInstance(input_lengths, np.ndarray)
        self.assertEqual(input_lengths.shape,
                         self.expected_input_lengths.shape)
        self.assertEqual(input_lengths.tolist(),
                         self.expected_input_lengths.tolist())

    def test_adapt_batch_returns_correct_label_lengths(self):
        adapter = CTCAdapter(self.padding_value)
        res_inp, _ = adapter.adapt_batch(self.seqs_in, self.seqs_out)

        label_lengths = res_inp[3]

        self.assertIsInstance(label_lengths, np.ndarray)
        self.assertEqual(label_lengths.tolist(),
                         self.expected_label_lengths.tolist())

    def test_adapt_batch_of_size_one_adds_no_padding(self):
        adapter = CTCAdapter(self.padding_value)

        seqs_in = [
            [[0, 1], [1, 2]]
        ]

        seqs_out = [[32, 51]]

        res_inp, _ = adapter.adapt_batch(seqs_in, seqs_out)
        x = res_inp[0]
        self.assertEqual(x.tolist(), seqs_in)

        labels = res_inp[1]
        self.assertEqual(x.tolist(), seqs_in)

        self.assertEqual(labels.tolist(), seqs_out)


class Seq2seqAdapterTests(TestCase):
    def setUp(self):
        self.seqs_in = [
            [[1, 1], [2, 2], [3, 3]],
            [[4, 4]]
        ]

        self.seqs_out = [
            [34, 85, 23],
            [28]
        ]

        char_table = CharacterTable()
        self.char_table = char_table
        self.start = char_table.encode(char_table.start)
        self.sentinel = char_table.encode(char_table.sentinel)
        self.adapter = Seq2seqAdapter(self.start, self.sentinel,
                                      num_classes=len(char_table))

    def test_adapt_batch_returns_valid_x(self):
        inputs, targets = self.adapter.adapt_batch(self.seqs_in, self.seqs_out)

        encoder_x, decoder_x = inputs
        self.assertTupleEqual(encoder_x.shape, (2, 3, 2))
        self.assertEqual(encoder_x.tolist(), [
            self.seqs_in[0],
            self.seqs_in[1] + [[0, 0], [0, 0]]
        ])

    def test_adapt_batch_returns_valid_decoder_inputs(self):
        inputs, targets = self.adapter.adapt_batch(self.seqs_in, self.seqs_out)

        encoder_x, decoder_x = inputs

        expected_decoder_x = [
            [self.start] + self.seqs_out[0],
            [self.start, self.seqs_out[1][0], self.sentinel, self.sentinel]
        ]

        expected_decoder_x = to_categorical(expected_decoder_x,
                                            num_classes=len(self.char_table))

        self.assertTupleEqual(decoder_x.shape, (2, 4, len(self.char_table)))
        self.assertEqual(decoder_x.tolist(), expected_decoder_x.tolist())

    def test_adapt_batch_returns_valid_targets(self):
        inputs, targets = self.adapter.adapt_batch(self.seqs_in, self.seqs_out)

        expected_targets = [
            self.seqs_out[0] + [self.sentinel],
            [self.seqs_out[1][0], self.sentinel, self.sentinel, self.sentinel]
        ]

        expected_targets = to_categorical(expected_targets,
                                          num_classes=len(self.char_table))

        self.assertTupleEqual(targets.shape, (2, 4, len(self.char_table)))
        self.assertEqual(targets.tolist(), expected_targets.tolist())


# todo: finish test that tests adapter when input is shorter than 2 * input_length + 1
# todo: cleanup ugly nested lists
