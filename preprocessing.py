from data import PreLoadedIterator


class ProcessingStep:
    def fit(self, batch):
        raise NotImplementedError

    def process(self, batch):
        raise NotImplementedError


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
    def fit(self, batch):
        pass

    def process(self, batch):
        return batch


class PreProcessor:
    def __init__(self):
        self._steps = []
        self._add_steps()

    def _add_steps(self):
        self._steps.append(SequencePadding())
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
