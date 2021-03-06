import numpy as np
from keras.utils import to_categorical


class ExampleAdapter:
    def adapt_input(self, xs):
        raise NotImplementedError

    def adapt_output(self, ys):
        raise NotImplementedError

    def adapt_batch(self, seqs_in, seqs_out):
        raise NotImplementedError


class EmptyListException(Exception):
    pass


class BadInputShapeException(Exception):
    pass


class CTCAdapter(ExampleAdapter):
    def __init__(self, y_padding):
        self._y_padding = y_padding

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
            s = list(seq)
            while len(s) < max_length:
                s = s + [self._y_padding]
            padded_seqs.append(s)

        return padded_seqs

    def _make_labels(self, seqs_out):
        sequence_length = len(seqs_out[0])
        labels = np.array(seqs_out, dtype=np.int32).reshape(-1, sequence_length)
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

        return [X, labels, input_lengths, label_lengths], labels


class Seq2seqAdapter(CTCAdapter):
    def __init__(self, start, sentinel, num_classes):
        super().__init__(y_padding=sentinel)

        self._start = start
        self._sentinel = sentinel
        self._num_classes = num_classes

    def adapt_batch(self, seqs_in, seqs_out):
        new_seqs_in = self._pad_input_sequences(seqs_in)
        new_seqs_out = self._pad_output_sequences(seqs_out)

        decoder_x = [[self._start] + s for s in new_seqs_out]

        decoder_x = to_categorical(decoder_x, num_classes=self._num_classes)

        targets = [s + [self._sentinel] for s in new_seqs_out]

        targets = to_categorical(targets, num_classes=self._num_classes)

        return [np.array(new_seqs_in), decoder_x], np.array(targets)
