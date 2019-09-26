from data.char_table import CharacterTable
from models.ctc_model import CtcModel
import os
from sources.compiled import CompilationSource
from train_ctc import LabelSource, CtcGenerator
from data.preprocessing import PreProcessor
from keras import layers
from config import CTCConfig
from train_ctc import build_model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./compiled')
    parser.add_argument('--max_examples', type=int, default=128)
    parser.add_argument('--cuda', type=bool, default=False)

    args = parser.parse_args()

    recurrent_layer = getattr(layers, args.recurrent_layer)

    num_features = CTCConfig().config_dict['num_features']
    num_cells = args.num_cells
    char_table = CharacterTable()
    model_path = args.model_path

    ctc_model = build_model(args.cuda, warp=False)

    data_path = args.data_path

    train_source = CompilationSource(
        os.path.join(data_path, 'train.h5py'), args.max_examples
    )

    test_source = CompilationSource(
        os.path.join(data_path, 'test.h5py'), args.max_examples
    )

    train_source = LabelSource(train_source, char_table)
    test_source = LabelSource(test_source, char_table)

    preprocessor = PreProcessor()

    train_gen = CtcGenerator(char_table, train_source, preprocessor, channels=num_features)
    test_gen = CtcGenerator(char_table, test_source, preprocessor, channels=num_features)

    model = ctc_model.compile_model(0.001)

    res = model.evaluate_generator(train_gen.get_examples(1), steps=len(train_gen))
    print('evalution on train data:', res)

    res = model.evaluate_generator(test_gen.get_examples(1), steps=len(test_gen))
    print('evalution on test data:', res)
