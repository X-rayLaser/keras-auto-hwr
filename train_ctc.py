from data.generators import BaseGenerator
from sources.preloaded import PreLoadedSource
from data.char_table import CharacterTable
from keras.callbacks import TensorBoard
from data.preprocessing import PreProcessor
import numpy as np
from sources.compiled import CompilationSource
from sources.iam_online import BadStrokeException
from keras import layers
from sources.wrappers import labels_source
from models.ctc_model import WarpCtcModel, CtcModel, MyCallback


def points_source(source, num_examples):
    sin = []
    sout = []
    for strokes, transcription in source.get_sequences():
        x0, y0, t0 = strokes[0].points[0]

        new_seq = []
        if len(sout) >= num_examples:
            break
        for stroke in strokes:
            points = []
            try:
                for x, y, t in stroke.points:
                    points.append((x - x0, y - y0, t - t0, 0))
            except BadStrokeException:
                pass
            points[-1] = points[-1][:-1] + (1, )
            new_seq.extend(points)

        sin.append(new_seq)
        sout.append(transcription)

    return PreLoadedSource(sin, sout)


class CtcGenerator(BaseGenerator):
    def __init__(self, mapping_table, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mapping = mapping_table

    def pad_seqsin(self, seqs_in):

        max_len = max(len(seq) for seq in seqs_in)

        res = []
        for seq in seqs_in:
            a = list(seq)
            while len(a) < max_len:
                a.append([0] * self._channels)
            res.append(a)

        return res

    def prepare_batch(self, seqs_in, seqs_out):
        m = len(seqs_out)

        labels = list(seqs_out)

        label_length = np.zeros([m, 1], dtype=np.int32)
        input_length = np.zeros([m, 1], dtype=np.int32)

        for i, seq in enumerate(seqs_out):
            label_length[i, 0] = len(seqs_out[i])
            input_length[i, 0] = len(seqs_in[i])

        if m > 1:
            max_len = max(len(row) for row in seqs_out)

            for i in range(len(labels)):
                while len(labels[i]) < max_len:
                    labels[i].append(self._mapping.encode(self._mapping.sentinel))

            seqs_in_pad = self.pad_seqsin(seqs_in)
            raise Exception('oops')

        seqs_in_pad = seqs_in

        n = len(seqs_in_pad[0])
        X = np.array(seqs_in_pad)

        X = X.reshape((m, n, self._channels))

        labels = np.array(labels, dtype=np.int32)

        return [X, labels, input_length, label_length], labels


def dummy_source():
    sin = 'HHHH    eee  lll  lll  ooo  ,,,  www   oooo  rrr   lll  ddd'
    sout = 'Hello, world'

    char_table = CharacterTable()

    codes = [char_table.encode(ch) for ch in sin]
    from keras.utils import to_categorical

    x = to_categorical(codes, num_classes=len(char_table))

    x = x.reshape(1, len(sin), -1)

    return PreLoadedSource(x, [sout])


class Normalizer:
    def __init__(self):
        self._mu = None
        self._sd = None

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


def normalized_source(source, normalizer):
    seqs_in = []
    seqs_out = []
    for seq_in, seq_out in source.get_sequences():
        seqs_in.append(seq_in)
        seqs_out.append(seq_out)

    processed = normalizer.preprocess(seqs_in)

    return PreLoadedSource(processed, seqs_out)


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./compiled')
    parser.add_argument('--max_examples', type=int, default=128)
    parser.add_argument('--lrate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warp', type=bool, default=False)
    parser.add_argument('--recurrent_layer', type=str, default='GRU')
    parser.add_argument('--num_cells', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='./weights/blstm/blstm.h5')

    args = parser.parse_args()

    print('training with following options:', args)

    compilation_train_source = CompilationSource(
        os.path.join(args.data_path, 'train.json')
    )

    compilation_validation_source = CompilationSource(
        os.path.join(args.data_path, 'validation.json')
    )

    char_table = CharacterTable()

    batch_size = 1
    embedding_size = 4
    num_train_examples = args.max_examples
    num_val_examples = max(1, num_train_examples // 2)
    label_space = len(char_table) + 1

    train_source = points_source(compilation_train_source, num_train_examples)
    val_source = points_source(compilation_validation_source, num_val_examples)

    normalizer = Normalizer()

    xs = [in_seq for in_seq, _ in train_source.get_sequences()]
    normalizer.fit(xs)

    train_source = normalized_source(train_source, normalizer)
    val_source = normalized_source(val_source, normalizer)

    preprocessor = PreProcessor()

    train_source = labels_source(train_source, char_table)
    val_source = labels_source(val_source, char_table)

    train_gen = CtcGenerator(char_table, train_source, preprocessor, channels=embedding_size)
    val_gen = CtcGenerator(char_table, val_source, preprocessor, channels=embedding_size)

    validation_steps = num_val_examples

    RNN_LAYER = getattr(layers, args.recurrent_layer)
    print('using layer', str(RNN_LAYER))

    if args.warp:
        ctc_model = WarpCtcModel(RNN_LAYER, label_space, embedding_size, num_cells=args.num_cells)
        ctc_model.fig_generator(train_gen, val_gen, args.lrate, args.epochs, char_table)
    else:
        ctc_model = CtcModel(RNN_LAYER, label_space,
                             embedding_size, num_cells=args.num_cells, save_path=args.save_path)

        ctc_model.fit_generator(train_gen, val_gen, args.lrate, args.epochs, char_table)


# todo: advanced preprocessing/normalizing stage
# todo: error correction by intepolating missing values and truncating too large point sequences
