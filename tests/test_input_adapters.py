from unittest import TestCase
import numpy as np
from keras.utils import to_categorical
from data.char_table import CharacterTable


class ExampleAdapter:
    def adapt_input(self, xs):
        pass

    def adapt_output(self, ys):
        pass

    def adapt_batch(self, seqs_in, seqs_out):
        pass


class EmptyListException(Exception):
    pass


class BadInputShapeException(Exception):
    pass


class CTCAdapter(ExampleAdapter):
    def __init__(self):
        self.character_table = CharacterTable()

    def _validate_shape(self, x, rank):
        msg = 'Input should be a list analog of rank {} numpy array.' \
              ' Got {}'.format(rank, x)
        try:
            a = np.array(x)
        except Exception:
            raise BadInputShapeException(msg)
        else:
            if len(a.shape) != rank:
                raise BadInputShapeException(msg)
            if a.shape[0] == 0 or a.shape[1] == 0:
                msg = 'All dimensions should be non zero. Got {}'.format(x)
                raise BadInputShapeException(msg)

    def _validate_single_input(self, xs):
        if len(xs) <= 0:
            raise EmptyListException()

        self._validate_shape(xs, rank=2)

    def _validate_batch_input(self, seqs_in):
        if len(seqs_in) <= 0:
            raise EmptyListException('seqs_in is empty')

        for s in seqs_in:
            self._validate_shape(s, rank=2)

        seqs_in = self._pad_input_sequences(seqs_in)
        self._validate_shape(seqs_in, rank=3)

    def adapt_input(self, xs):
        self._validate_single_input(xs)
        sequence_length = len(xs)
        num_features = len(xs[0])

        return np.array(xs).reshape((1, sequence_length, num_features))

    def _max_length(self, sequences):
        return max([len(seq) for seq in sequences])

    def _pad_input_sequences(self, seqs_in, padding_value=0):
        max_length = self._max_length(seqs_in)
        num_features = len(seqs_in[0][0])

        new_input_seq = []
        for seq in seqs_in:
            s = list(seq)
            while len(s) < max_length:
                s.append([padding_value] * num_features)
            new_input_seq.append(s)

        return new_input_seq

    def _pad_output_sequences(self, seqs_out):
        max_length = self._max_length(seqs_out)

        padded_seqs = []
        for seq in seqs_out:
            s = str(seq)
            while len(s) < max_length:
                s += self.character_table.sentinel
            padded_seqs.append(s)

        return padded_seqs

    def _make_labels(self, seqs_out):
        labels = []
        for s in seqs_out:
            seq = [self.character_table.encode(ch) for ch in s]
            labels.append(seq)

        sequence_length = len(labels[0])
        alphabet_size = len(self.character_table)
        labels = to_categorical(labels, num_classes=alphabet_size)
        labels = labels.reshape(-1, sequence_length, alphabet_size)
        return labels

    def _make_lengths(self, sequences):
        batch_size = len(sequences)
        a = np.array([len(s) for s in sequences], dtype=np.int32)
        return a.reshape(batch_size, 1)

    def adapt_batch(self, seqs_in, seqs_out):
        self._validate_batch_input(seqs_in)

        new_input_seq = self._pad_input_sequences(seqs_in)
        new_output_seq = self._pad_output_sequences(seqs_out)

        X = np.array(new_input_seq)

        labels = self._make_labels(new_output_seq)

        input_lengths = self._make_lengths(seqs_in)
        label_lengths = self._make_lengths(seqs_out)

        return [X, labels, input_lengths, label_lengths], seqs_out


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
