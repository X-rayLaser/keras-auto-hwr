from data.char_table import CharacterTable
from models.ctc_model import CtcModel
import os
from sources.compiled import CompilationSource
from train_ctc import LabelSource, CtcGenerator
from data.preprocessing import PreProcessor
from keras import layers


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--recurrent_layer', type=str, default='LSTM')
    parser.add_argument('--num_cells', type=int, default=100)
    parser.add_argument('--max_examples', type=int, default=128)

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

    data_path = './compiled'

    train_source = CompilationSource(
        os.path.join(data_path, 'train.h5py'), args.max_examples
    )

    test_source = CompilationSource(
        os.path.join(data_path, 'test.h5py'), args.max_examples
    )

    train_source = LabelSource(train_source, char_table)
    test_source = LabelSource(test_source, char_table)

    preprocessor = PreProcessor()

    train_gen = CtcGenerator(char_table, train_source, preprocessor, channels=embedding_size)
    test_gen = CtcGenerator(char_table, test_source, preprocessor, channels=embedding_size)

    model = ctc_model.compile_model(0.001)

    res = model.evaluate_generator(train_gen.get_examples(1), steps=len(train_gen))
    print('evalution on train data:', res)

    res = model.evaluate_generator(test_gen.get_examples(1), steps=len(train_gen))
    print('evalution on test data:', res)
