import numpy as np
from keras.utils import to_categorical


class DataSetGenerator:
    def __init__(self, lines_iterator, char_table, pre_processor):
        self._char_table = char_table
        self._iter = lines_iterator
        self._pre_processor = pre_processor

    def prepare_batch(self, hand_writings, transcriptions):
        x = np.array(hand_writings)

        x_norm = x.reshape((-1, x.shape[1], 1))

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

    def _raw_batches(self, batch_size):
        while True:
            hand_writings = []
            transcriptions = []
            for handwriting, transcription in self._iter.get_sequences():
                hand_writings.append(handwriting)
                transcriptions.append(transcription)

                if len(hand_writings) >= batch_size:
                    yield hand_writings, transcriptions

                    hand_writings = []
                    transcriptions = []

            if len(hand_writings) > 0:
                yield hand_writings, transcriptions

    def get_examples(self, batch_size=64):
        for hand_writings, transcriptions in self._raw_batches(batch_size):
            hand_writings, transcriptions = self._pre_processor.process(
                hand_writings, transcriptions
            )
            yield self.prepare_batch(hand_writings, transcriptions)

        # todo: separate generator for images instead of strokes

    def __len__(self):
        return len(self._iter)


class AttentionModelDataGenerator(DataSetGenerator):
    def __init__(self, lines_iterator, char_table, pre_processor, Tx, Ty, encoder_states, channels):
        super().__init__(lines_iterator, char_table, pre_processor)

        self._Tx = Tx
        self._Ty = Ty
        self._encoder_states = encoder_states
        self._channels = channels

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