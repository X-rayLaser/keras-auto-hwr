from sources.preloaded import PreLoadedSource
from data.char_table import CharacterTable
from keras import layers
from models.ctc_model import WarpCtcModel, CtcModel
from api import CompilationHome, create_source
from data.generators import MiniBatchGenerator
from data.example_adapters import CTCAdapter
from config import CTCConfig
from keras.utils import to_categorical
from data.data_set_home import DataSetHome


def dummy_source():
    sin = 'HHHH    eee  lll  lll  ooo  ,,,  www   oooo  rrr   lll  ddd'
    sout = 'Hello, world'

    char_table = CharacterTable()

    codes = [char_table.encode(ch) for ch in sin]

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

    location = CompilationHome('ds1').root_dir
    ds_home = DataSetHome(location, create_source)

    train_source, val_source, test_slice = ds_home.get_slices()

    adapter = CTCAdapter()

    train_gen = MiniBatchGenerator(train_source, adapter, batch_size=1)
    val_gen = MiniBatchGenerator(val_source, adapter, batch_size=1)

    ctc_model = build_model(args.cuda, args.warp)
    ctc_model.fit_generator(train_gen, val_gen, args.lrate, args.epochs, char_table)


# todo: reuse code for predictor, output decoder etc
# todo: advanced preprocessing/normalizing stage
# todo: error correction by intepolating missing values and truncating too large point sequences
