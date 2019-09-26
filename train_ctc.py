from data.generators import BaseGenerator
from sources.preloaded import PreLoadedSource
from data.char_table import CharacterTable
from data.preprocessing import PreProcessor
import numpy as np
from sources.compiled import CompilationSource
from keras import layers
from sources.wrappers import LabelSource, Normalizer
from models.ctc_model import WarpCtcModel, CtcModel
from config import CTCConfig


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

        if m > 1:
            max_len = max(len(row) for row in seqs_out)

            for i in range(len(labels)):
                while len(labels[i]) < max_len:
                    labels[i].append(self._mapping.encode(self._mapping.sentinel))

            seqs_in_pad = self.pad_seqsin(seqs_in)
        else:
            seqs_in_pad = seqs_in

        n = len(seqs_in_pad[0])
        X = np.array(seqs_in_pad)

        X = X.reshape((m, n, self._channels))

        labels = np.array(labels, dtype=np.int32)

        label_length = np.zeros([m, 1], dtype=np.int32)
        input_length = np.zeros([m, 1], dtype=np.int32)

        for i in range(len(labels)):
            label_length[i, 0] = len(labels[i])
            input_length[i, 0] = len(seqs_in_pad[i])

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


def normalized_source(source, normalizer):
    seqs_in = []
    seqs_out = []
    for seq_in, seq_out in source.get_sequences():
        seqs_in.append(seq_in)
        seqs_out.append(seq_out)

    processed = normalizer.preprocess(seqs_in)

    return PreLoadedSource(processed, seqs_out)


def build_model(cuda, warp):
    from config import CTCConfig

    ctc_config = CTCConfig()
    rnn_layer = ctc_config.config_dict['recurrent_layer']
    num_cells = ctc_config.config_dict['num_cells']
    weights_location = ctc_config.config_dict['weights_location']
    num_features = ctc_config.config_dict['num_features']

    if rnn_layer == 'SimpleRNN':
        RNN_LAYER = getattr(layers, rnn_layer)
    else:
        if cuda and rnn_layer == 'LSTM':
            RNN_LAYER = getattr(layers, 'CuDNNLSTM')
        elif cuda and rnn_layer == 'GRU':
            RNN_LAYER = getattr(layers, 'CuDNNGRU')
        else:
            RNN_LAYER = getattr(layers, rnn_layer)

    if warp:
        ctc_model = WarpCtcModel(RNN_LAYER, num_features, num_cells=num_cells)
    else:
        ctc_model = CtcModel(RNN_LAYER,
                             num_features, num_cells=num_cells, save_path=weights_location)

    return ctc_model


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./compiled')
    parser.add_argument('--max_examples', type=int, default=1)

    parser.add_argument('--lrate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)

    parser.add_argument('--warp', type=bool, default=False)
    parser.add_argument('--cuda', type=bool, default=False)

    args = parser.parse_args()

    print('training with following options:', args)

    char_table = CharacterTable()
    num_features = CTCConfig().config_dict['num_features']
    num_train_examples = args.max_examples
    num_val_examples = max(1, num_train_examples // 2)

    train_source = CompilationSource(
        os.path.join(args.data_path, 'train.h5py'),
        num_train_examples
    )

    val_source = CompilationSource(
        os.path.join(args.data_path, 'validation.h5py'),
        num_val_examples
    )

    preprocessor = PreProcessor()

    train_source = LabelSource(train_source, char_table)
    val_source = LabelSource(val_source, char_table)

    train_gen = CtcGenerator(char_table, train_source, preprocessor, channels=num_features)
    val_gen = CtcGenerator(char_table, val_source, preprocessor, channels=num_features)

    ctc_model = build_model(args.cuda, args.warp)
    ctc_model.fit_generator(train_gen, val_gen, args.lrate, args.epochs, char_table)


# todo: advanced preprocessing/normalizing stage
# todo: error correction by intepolating missing values and truncating too large point sequences
