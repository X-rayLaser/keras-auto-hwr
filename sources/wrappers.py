import numpy as np

from sources import BaseSource
from sources.preloaded import PreLoadedSource


class WordsSource(BaseSource):
    def __init__(self, source):
        self._source = source

    def __len__(self):
        return len(self._source)

    def _remove_apostrpohs(self, seq):
        res = ''.join(seq.split('&apos;'))
        res = ''.join(res.split('&quot;'))
        return res

    def _clean(self, seq):
        s = ''
        for ch in seq.strip():
            if ch.isalpha():
                s += ch

        return s

    def get_sequences(self):
        for seq_in, transcription in self._source.get_sequences():
            transcription = self._remove_apostrpohs(transcription)

            words = [self._clean(word) for word in transcription.split(' ')]

            yield seq_in, words


def embeddings_source(source, num_examples):
    from train_on_embeddings import auto_encoder, get_embeddings
    embeddings, transcriptions, _, _ = get_embeddings(auto_encoder.get_encoder(), source, num_examples)
    return PreLoadedSource(embeddings, transcriptions)


class LabelSource(BaseSource):
    def __init__(self, source, mapping_table):
        self._source = source
        self._mapping_table = mapping_table

    def __len__(self):
        return len(self._source)

    def get_sequences(self):
        for seq_in, seq_out in self._source.get_sequences():
            label_seq = [self._mapping_table.encode(ch) for ch in seq_out]

            yield seq_in, label_seq


class CTCAdaptedSource(BaseSource):
    def __init__(self, source, padding_value=0):
        self._source = source
        self._padding = padding_value

    def __len__(self):
        return len(self._source)

    def get_sequences(self):
        for seq_in, seq_out in self._source.get_sequences():
            seqs_in_pad = list(seq_in)

            while len(seqs_in_pad) <= 2 * len(seq_out) + 1:
                n = len(seqs_in_pad[0])
                seqs_in_pad.append([self._padding] * n)

            yield seqs_in_pad, seq_out


class Normalizer:
    def __init__(self):
        self._mu = None
        self._sd = None

    def set_mean(self, mu):
        self._mu = mu

    def set_deviation(self, sd):
        self._sd = sd

    @property
    def mu(self):
        return self._mu

    @property
    def sd(self):
        return self._sd

    def fit(self, X):
        sequence = []
        for x in X:
            sequence.extend(x)

        self._mu = np.mean(sequence, axis=0)
        self._sd = np.std(sequence, axis=0)

    def preprocess(self, X):
        res = []
        for x in X:
            x_norm = (x - self._mu) / self._sd

            # we do not want to normalize END-OF-STROKE flag which is last in the tuple
            x_norm[:, -1] = np.array(x)[:, -1]
            res.append(x_norm.tolist())

        return res


class OffsetPointsSource(BaseSource):
    def __init__(self, source):
        self._source = source

    def __len__(self):
        return len(self._source)

    def get_sequences(self):
        for strokes, transcription in self._source.get_sequences():
            x0, y0, t0 = strokes[0].points[0]

            new_seq = []
            for stroke in strokes:
                points = []

                for x, y, t in stroke.points:
                    points.append((x - x0, y - y0, t - t0, 0))

                points[-1] = points[-1][:-1] + (1,)
                new_seq.extend(points)

            yield new_seq, transcription


class NormalizedSource(BaseSource):
    def __init__(self, source, normalizer):
        self._source = source
        self._normalizer = normalizer

    def __len__(self):
        return len(self._source)

    def get_sequences(self):
        for points, transcription in self._source.get_sequences():
            norm = self._normalizer.preprocess([points])[0]
            yield norm, transcription