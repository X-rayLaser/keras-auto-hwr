from data import PreLoadedIterator
import numpy as np


class ProcessingStep:
    def fit(self, batch):
        raise NotImplementedError

    def process(self, batch):
        raise NotImplementedError


class SignalMaker(ProcessingStep):
    def fit(self, batch):
        pass

    def process(self, batch):
        input_seqs = []
        target_seqs = []
        for points_line, transcription in batch.get_lines():
            heights = [y for x, y in points_line]
            input_seqs.append(heights)
            target_seqs.append(transcription)
        return PreLoadedIterator(input_seqs, target_seqs)


class SequencePadding(ProcessingStep):
    def __init__(self, input_padding=0, target_padding=' '):
        self._input_len = 0
        self._output_len = 0
        self._input_pad = input_padding
        self._target_pad = target_padding

    def fit(self, batch):
        for handwriting, transcription in batch.get_lines():
            self._input_len = max(len(handwriting), self._input_len)
            self._output_len = max(len(transcription), self._output_len)

    def process(self, batch):
        hand_writings = []
        transcriptions = []
        for hand_writing, transcription in batch.get_lines():
            hwr = list(hand_writing)
            hwr = self._pad_input(hwr)
            transcription = self._pad_target(transcription)

            hand_writings.append(hwr)
            transcriptions.append(transcription)

        return PreLoadedIterator(hand_writings, transcriptions)

    def _pad_input(self, input_seq):
        while len(input_seq) < self._input_len:
            input_seq.append(self._input_pad)

        if len(input_seq) > self._input_len:
            input_seq = input_seq[:self._input_len]

        return input_seq

    def _pad_target(self, target_seq):
        while len(target_seq) < self._output_len:
            target_seq += self._target_pad

        return target_seq


class PrincipalComponentAnalysis(ProcessingStep):
    def fit(self, batch):
        pass

    def process(self, batch):
        return batch


class Normalization(ProcessingStep):
    def __init__(self):
        self._mu = None
        self._std = None

    def fit(self, batch):

        hand_writings = []
        for points, transcription in batch.get_lines():
            hand_writings.append(points)

        self._mu = np.mean(hand_writings, axis=0)
        self._std = np.std(hand_writings, axis=0)

    def remove_constant_columns(self, a, epsilon=0.0001):
        all_column_indices = np.arange(len(a.shape[1]))
        column_indices = all_column_indices[self._std > epsilon]
        return a[:, column_indices]

    def process(self, batch):
        hand_writings = []
        transcriptions = []
        for points, transcription in batch.get_lines():
            hand_writings.append(points)
            transcriptions.append(transcription)

        epsilon = 0.001
        a = (np.array(hand_writings) - self._mu) / (self._std + epsilon)

        return PreLoadedIterator(a.tolist(), transcriptions)


class PreProcessor:
    def __init__(self, char_table):
        self._steps = []
        self._char_table = char_table
        self._add_steps()


    def _add_steps(self):
        self._steps.append(SignalMaker())
        self._steps.append(SequencePadding(target_padding=self._char_table.sentinel))
        self._steps.append(PrincipalComponentAnalysis())
        self._steps.append(Normalization())

    def fit(self, batch):
        for step in self._steps:
            step.fit(batch)
            batch = step.process(batch)

    def process(self, batch):
        for step in self._steps:
            batch = step.process(batch)

        return batch
