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


class CTCAdapter(ExampleAdapter):
    def adapt_input(self, xs):
        Tx = len(xs)
        n = len(xs[0])
        return np.array(xs).reshape((1, Tx, n))

    def _pad_input_sequences(self, seqs_in):
        max_length = max([len(seq) for seq in seqs_in])
        n = len(seqs_in[0][0])

        new_input_seq = []
        for seq in seqs_in:
            s = list(seq)
            while len(s) < max_length:
                s.append([0] * n)
            new_input_seq.append(s)

        return new_input_seq

    def _pad_output_sequences(self, seqs_out):
        max_length = max([len(seq) for seq in seqs_out])

        padded_seqs = []
        char_table = CharacterTable()
        for seq in seqs_out:
            s = str(seq)
            while len(s) < max_length:
                s += char_table.sentinel
            padded_seqs.append(s)

        return padded_seqs

    def _encode_output_sequences(self, seqs_out):
        char_table = CharacterTable()

        labels = []
        for s in seqs_out:
            seq = [char_table.encode(ch) for ch in s]
            labels.append(seq)
        return labels

    def adapt_batch(self, seqs_in, seqs_out):
        new_input_seq = self._pad_input_sequences(seqs_in)
        new_output_seq = self._pad_output_sequences(seqs_out)

        X = np.array(new_input_seq)

        char_table = CharacterTable()

        labels = self._encode_output_sequences(new_output_seq)

        Ty = len(labels[0])
        labels = to_categorical(labels, num_classes=len(char_table))
        labels = labels.reshape(-1, Ty, len(char_table))

        input_lengths = np.zeros((len(seqs_out), 1))
        label_lengths = np.zeros_like(input_lengths)

        for i in range(len(seqs_out)):
            input_lengths[i, 0] = len(new_input_seq[i])
            label_lengths[i, 0] = len(seqs_out[i])

        inputs = [X, labels, input_lengths, label_lengths]
        outputs = seqs_out
        return inputs, outputs


class CTCAdapterTests(TestCase):
    def setUp(self):
        seqs_in = [
            [[0, 1], [1, 2]],
            [[2, 4]]
        ]
        seqs_out = ['A', 'bc']
        self.seqs_in = seqs_in
        self.seqs_out = seqs_out

    def test_adapt_input(self):
        adapter = CTCAdapter()

        xs = [[0, 1, 3], [1, 2, 4]]
        res = adapter.adapt_input(xs)
        expected_shape = np.array([xs]).shape
        expected_res = np.array(xs).reshape(expected_shape)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape, expected_shape)
        self.assertEqual(res.tolist(), expected_res.tolist())

    def test_adapt_batch_returns_correct_X(self):
        adapter = CTCAdapter()

        expected_X = np.array([
            self.seqs_in[0],
            self.seqs_in[1] + [[0, 0]]
        ])

        res_inp, _ = adapter.adapt_batch(self.seqs_in, self.seqs_out)

        X = res_inp[0]

        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.shape, (2, 2, 2))
        self.assertEqual(X.tolist(), expected_X.tolist())

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
        char_table = CharacterTable()
        expected_labels = [
            [char_table.encode('A'), char_table.encode(char_table.sentinel)],
            [char_table.encode('b'), char_table.encode('c')]
        ]

        expected_labels = to_categorical(expected_labels, len(char_table)).reshape(2, 2, len(char_table))
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(labels.tolist(), expected_labels.tolist())

    def test_adapt_batch_returns_correct_input_lengths(self):
        adapter = CTCAdapter()
        res_inp, _ = adapter.adapt_batch(self.seqs_in, self.seqs_out)

        input_lengths = res_inp[2]
        expected_lengths = np.array([[2], [2]])

        self.assertIsInstance(input_lengths, np.ndarray)
        self.assertEqual(input_lengths.shape, expected_lengths.shape)
        self.assertEqual(input_lengths.tolist(), expected_lengths.tolist())

    def test_adapt_batch_returns_correct_label_lengths(self):
        adapter = CTCAdapter()
        res_inp, _ = adapter.adapt_batch(self.seqs_in, self.seqs_out)

        label_lengths = res_inp[3]
        expected_lengths = np.array([[1], [2]])

        self.assertIsInstance(label_lengths, np.ndarray)
        self.assertEqual(label_lengths.tolist(), expected_lengths.tolist())



# todo: consider removing padding for both input and output sequences
# todo: test more edge cases
# todo: clean up
