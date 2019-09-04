from sources.preloaded import PreLoadedSource
import numpy as np


class ProcessingStep:
    def fit(self, hand_writings, transcriptions):
        raise NotImplementedError

    def process(self, hand_writings, transcriptions):
        raise NotImplementedError


class Flattening(ProcessingStep):
    def fit(self, hand_writings, transcriptions):
        pass

    def process(self, hand_writings, transcriptions):
        points_seq = []
        for i in range(len(transcriptions)):
            strokes = hand_writings[i]
            points = []
            for stroke in strokes:
                points.extend(stroke.points)

            points_seq.append(points)

        return points_seq, transcriptions


class SignalMaker(ProcessingStep):
    def fit(self, hand_writings, transcriptions):
        pass

    def process(self, hand_writings, transcriptions):
        input_seqs = []

        for i in range(len(hand_writings)):
            points_line = hand_writings[i]
            heights = [y for x, y in points_line]
            input_seqs.append(heights)
        return input_seqs, transcriptions


class DeltaSignal(ProcessingStep):
    def fit(self,  hand_writings, transcriptions):
        pass

    def to_deltas(self, heights):
        a = np.array(heights)
        deltas = a[1:] - a[:-1]
        return deltas.tolist()

    def process(self,  hand_writings, transcriptions):
        input_seqs = []

        for i in range(len(hand_writings)):
            points = hand_writings[i]
            xs = [x for x, y in points]
            ys = [y for x, y in points]

            #xs = self.to_deltas(xs)
            #ys = self.to_deltas(ys)
            input_seqs.append(xs + ys)

        return input_seqs, transcriptions


class SequencePadding(ProcessingStep):
    def __init__(self, input_padding=0, target_padding=' ', input_len=None, output_len=None):
        self._input_len = input_len
        self._output_len = output_len
        self._input_pad = input_padding
        self._target_pad = target_padding

    def fit(self, hand_writings, transcriptions):
        if self._input_len is None or self._output_len is None:
            self._input_len = 0
            self._output_len = 0
            for i in range(len(hand_writings)):
                handwriting = hand_writings[i]
                transcription = transcriptions[i]
                self._input_len = max(len(handwriting), self._input_len)
                self._output_len = max(len(transcription), self._output_len)

    def process(self, hand_writings, transcriptions):
        input_seq = []
        output_seq = []
        for i in range(len(hand_writings)):
            hand_writing = hand_writings[i]
            transcription = transcriptions[i]
            hwr = list(hand_writing)
            hwr = self._pad_input(hwr)
            transcription = self._pad_target(transcription)

            input_seq.append(hwr)
            output_seq.append(transcription)

        return input_seq, output_seq

    def _pad_input(self, input_seq):
        while len(input_seq) < self._input_len:
            input_seq.append(self._input_pad)

        if len(input_seq) > self._input_len:
            input_seq = input_seq[:self._input_len]

        return input_seq

    def _pad_target(self, target_seq):
        while len(target_seq) < self._output_len:
            target_seq += self._target_pad

        if len(target_seq) > self._output_len:
            target_seq = target_seq[:self._output_len]

        return target_seq


class Truncation(ProcessingStep):
    def __init__(self, fraction=0.5):
        self._fraction = fraction
        self._max_input_len = 0
        self._max_output_len = 0

    def fit(self, hand_writings, transcriptions):

        inp_len = max([len(inp) for inp in hand_writings])
        self._max_input_len = inp_len

        output_len = max([len(output) for output in transcriptions])
        self._max_output_len = output_len

    def _calculate_cutoff(self, max_len):
        return int(round(max_len * self._fraction))

    def process(self, hand_writings, transcriptions):
        inputs = []
        outputs = []
        input_len = self._calculate_cutoff(self._max_input_len)
        output_len = self._calculate_cutoff(self._max_output_len)

        for i in range(len(hand_writings)):
            input_seq = hand_writings[i]
            output_seq = transcriptions[i]
            inputs.append(input_seq[:input_len])
            outputs.append(output_seq[:output_len])

        return inputs, outputs


class DftCompress(ProcessingStep):
    def __init__(self, block_size=64, cutoff=16):
        self._block_size = block_size
        self._cutoff = cutoff

    def fit(self, batch):
        pass

    def compress(self, point_seq):
        while len(point_seq) % self._block_size != 0:
            point_seq.append(0)

        res = []
        for i in range(0, len(point_seq), self._block_size):
            block = point_seq[i:i + self._block_size]
            spectrum = np.fft.fft(block)
            spectrum = spectrum[:self._cutoff]
            res.extend(np.abs(spectrum).tolist())

        return res

    def process(self, batch):
        spectrums = []
        transcriptions = []
        for handwriting, transcription in batch.get_sequences():
            spectrums.append(self.compress(handwriting))
            transcriptions.append(transcription)

        return PreLoadedSource(spectrums, transcriptions)


class PrincipalComponentAnalysis(ProcessingStep):
    def fit(self, hand_writings, transcriptions):
        pass

    def process(self, hand_writings, transcriptions):
        pass


class StreamSplit(ProcessingStep):
    def fit(self, hand_writings, transcriptions):
        pass

    def process(self, hand_writings, transcriptions):
        inp_seqs = []
        for points in hand_writings:
            if len(points) % 2 != 0:
                points = points[:-1]

            it = iter(points)
            inp_seqs.append(list(zip(it, it)))

        return inp_seqs, transcriptions


class Normalization(ProcessingStep):
    def __init__(self):
        self._mu = None
        self._std = None

    def fit(self, hand_writings, transcriptions):
        self._mu = np.mean(hand_writings, axis=0)
        self._std = np.std(hand_writings, axis=0)

    def remove_constant_columns(self, a, epsilon=0.0001):
        all_column_indices = np.arange(len(a.shape[1]))
        column_indices = all_column_indices[self._std > epsilon]
        return a[:, column_indices]

    def process(self, hand_writings, transcriptions):
        epsilon = 0.001
        a = (np.array(hand_writings) - self._mu) / (self._std + epsilon)

        return a.tolist(), transcriptions


class PreProcessor:
    def __init__(self, char_table):
        self._steps = []
        self._char_table = char_table

    def add_step(self, step):
        self._steps.append(step)

    def fit(self, hand_writings, transcriptions):
        for step in self._steps:
            step.fit(hand_writings, transcriptions)
            hand_writings, transcriptions = step.process(hand_writings, transcriptions)

    def process(self, hand_writings, transcriptions):
        for step in self._steps:
            hand_writings, transcriptions = step.process(hand_writings, transcriptions)

        return hand_writings, transcriptions
