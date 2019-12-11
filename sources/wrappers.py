import numpy as np
from sources import BaseSource
from sources.base import BaseSourceWrapper
from sources.preloaded import PreLoadedSource
import json


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

    @staticmethod
    def from_json(path):
        with open(path, 'r') as f:
            s = f.read()

        d = json.loads(s)
        normalizer = Normalizer()
        mu = np.array(d['mu'])
        sd = np.array(d['sd'])
        normalizer.set_mean(mu)
        normalizer.set_deviation(sd)
        return normalizer

    def to_json(self, path):
        d = {
            'mu': np.array(self.mu).tolist(),
            'sd': np.array(self.sd).tolist()
        }
        with open(path, 'w') as f:
            f.write(json.dumps(d))

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


class DenormalizedSource(BaseSource):
    def __init__(self, source, normalizer):
        self._source = source
        self._normalizer = normalizer

    def __len__(self):
        return len(self._source)

    def get_sequences(self):
        mu = self._normalizer.mu
        sd = self._normalizer.sd

        for points, transcription in self._source.get_sequences():

            denormalized = [(p * sd + mu).tolist() for p in points]

            for i, p in enumerate(denormalized):
                p[3] = points[i][3]

            yield denormalized, transcription


class H5pySource(BaseSource):
    def __init__(self, h5py_ds, random_order=True):
        self._h5py = h5py_ds
        self._random = random_order

    def __len__(self):
        return len(self._h5py)

    def get_sequences(self):
        return self._h5py.get_data(random_order=self._random)


class PreprocessedSource(BaseSourceWrapper):
    def __init__(self, source, preprocessor):
        super().__init__(source)
        self._preprocessor = preprocessor

    def get_sequences(self):
        for xs, ys in self._source.get_sequences():
            yield self._preprocessor.pre_process_example(xs, ys)


class ConstrainedSource(BaseSourceWrapper):
    def __init__(self, source, num_lines):
        super().__init__(source)
        self._num_lines = num_lines

        self._use_all = (num_lines == 0)

    def get_sequences(self):
        for j, (seq_in, seq_out) in enumerate(self._source.get_sequences()):
            #print(j, seq_out)
            if j % 500 == 0:
                print('Fetched {} examples'.format(j))

            if j >= self._num_lines and not self._use_all:
                break
            yield seq_in, seq_out


class PlainListSource(BaseSourceWrapper):
    def get_sequences(self):
        for strokes, t in self._source.get_sequences():
            points = [stroke.points for stroke in strokes]
            yield points, t
