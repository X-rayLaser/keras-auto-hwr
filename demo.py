from data.char_table import CharacterTable
from models.ctc_model import CtcModel
from models.ctc_model import MyCallback
import os
from sources.compiled import CompilationSource
from train_ctc import LabelSource, CtcGenerator
from data.preprocessing import PreProcessor
from keras import layers


def model_demo(model, char_table):
    data_path = './compiled'

    train_source = CompilationSource(
        os.path.join(data_path, 'train.h5py'), 10
    )

    test_source = CompilationSource(
        os.path.join(data_path, 'test.h5py'), 10
    )

    train_source = LabelSource(train_source, char_table)
    test_source = LabelSource(test_source, char_table)

    preprocessor = PreProcessor()
    train_gen = CtcGenerator(char_table, train_source, preprocessor, channels=embedding_size)
    test_gen = CtcGenerator(char_table, test_source, preprocessor, channels=embedding_size)

    callback = MyCallback(model, train_gen, test_gen, char_table)
    callback.demo(train_gen)
    callback.demo(test_gen)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--recurrent_layer', type=str, default='LSTM')
    parser.add_argument('--num_cells', type=int, default=100)
    parser.add_argument('--model_path', type=str, default='./weights/blstm/blstm.h5')

    args = parser.parse_args()

    recurrent_layer = getattr(layers, args.recurrent_layer)

    embedding_size = 4
    num_cells = args.num_cells
    char_table = CharacterTable()
    model_path = args.model_path

    label_space = len(char_table) + 1

    ctc_model = CtcModel(recurrent_layer, label_space, embedding_size=embedding_size,
                         num_cells=num_cells, save_path=model_path)

    model_demo(ctc_model.inference_model, char_table)
