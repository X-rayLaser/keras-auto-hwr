import numpy as np
from keras.utils import to_categorical


class BaseGenerator:
    def __init__(self, strokes_iterator, pre_processor, channels):
        self._iter = strokes_iterator
        self._pre_processor = pre_processor
        self._channels = channels

    def prepare_batch(self, seqs_in, seqs_out):
        raise NotImplementedError

    def _raw_batches(self, batch_size):
        while True:
            seqs_in = []
            seqs_out = []
            for handwriting, transcription in self._iter.get_sequences():
                seqs_in.append(handwriting)
                seqs_out.append(transcription)

                if len(seqs_in) >= batch_size:
                    yield seqs_in, seqs_out

                    seqs_in = []
                    seqs_out = []

            if len(seqs_in) > 0:
                yield seqs_in, seqs_out

    def get_examples(self, batch_size=64):
        for seqs_in, seqs_out in self._raw_batches(batch_size):
            seqs_in, seqs_out = self._pre_processor.process(
                seqs_in, seqs_out
            )
            yield self.prepare_batch(seqs_in, seqs_out)

    def __len__(self):
        return len(self._iter)


class AutoEncoderGenerator(BaseGenerator):
    def prepare_batch(self, seqs_in, seqs_out):
        x = np.array(seqs_in)

        x = x.reshape((-1, x.shape[1], 2))
        #x_noisy = x + np.random.normal(loc=0, scale=10, size=x.shape)
        x_noisy = x
        y_in = x[:, :-1, :]
        y_out = x[:, 1:, :]

        return [x_noisy, y_in], y_out


class DataSetGenerator(BaseGenerator):
    def __init__(self, lines_iterator, char_table, pre_processor, channels):
        super().__init__(lines_iterator, pre_processor, channels)
        self._char_table = char_table

    def prepare_batch(self, hand_writings, transcriptions):
        x = np.array(hand_writings)

        x_norm = x.reshape((-1, x.shape[1], self._channels))

        char_table = self._char_table
        seqs = []

        for tok in transcriptions:
            s = char_table.start + tok + char_table.sentinel
            encoded = [char_table.encode(ch) for ch in s]
            seqs.append(encoded)

        padded_seqs = np.array(seqs)

        y = to_categorical(padded_seqs, num_classes=len(char_table))
        y_in = y[:, :-1, :]
        y_out = y[:, 1:, :]

        return [x_norm, y_in], y_out


class AttentionModelDataGenerator(DataSetGenerator):
    def __init__(self, lines_iterator, char_table, pre_processor, Tx, Ty, encoder_states, channels):
        super().__init__(lines_iterator, char_table, pre_processor, channels)

        self._Tx = Tx
        self._Ty = Ty
        self._encoder_states = encoder_states

    def prepare_batch(self, hand_writings, transcriptions):
        batch_size = len(transcriptions)
        alphabet_size = len(self._char_table)

        x = np.array(hand_writings)

        x_norm = x.reshape((-1, x.shape[1], self._channels))
        initial_state = np.zeros((batch_size, self._encoder_states))

        targets = []
        for tok in transcriptions:
            s = tok + self._char_table.sentinel

            encoded = [self._char_table.encode(ch) for ch in s]
            targets.append(encoded)

        targets = np.array(targets)
        y = to_categorical(targets, num_classes=alphabet_size)

        final_y = []
        for t in range(self._Ty):
            final_y.append(y[:, t, :])

        char_table = self._char_table
        initial_y = [char_table.encode(char_table.start) for _ in range(batch_size)]

        initial_y = to_categorical(
            initial_y, num_classes=alphabet_size
        ).reshape((batch_size, 1, alphabet_size))

        self._char_table.encode(self._char_table.start)
        return [x_norm, initial_state, initial_y], final_y

    def debug_examples(self):
        for hand_writings, transcriptions in self._raw_batches(1):
            hwr, t = self._pre_processor.process(hand_writings, transcriptions)
            batch = self.prepare_batch(hwr, t)
            yield hand_writings, batch


class MiniBatchGenerator:
    def __init__(self, source, example_adapter, batch_size=16):
        self._source = source
        self._example_adapter = example_adapter
        self._batch_size = batch_size

    def prepare_batch(self, seqs_in, seqs_out):
        return self._example_adapter.adapt_batch(seqs_in, seqs_out)

    def _raw_batches(self, batch_size):
        while True:
            seqs_in = []
            seqs_out = []
            for handwriting, transcription in self._source.get_sequences():
                seqs_in.append(handwriting)
                seqs_out.append(transcription)

                if len(seqs_in) >= batch_size:
                    yield seqs_in, seqs_out

                    seqs_in = []
                    seqs_out = []

            if len(seqs_in) > 0:
                yield seqs_in, seqs_out

    def get_examples(self):
        for seqs_in, seqs_out in self._raw_batches(self._batch_size):
            yield self.prepare_batch(seqs_in, seqs_out)

    def __len__(self):
        return len(self._source)
