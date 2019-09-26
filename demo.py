from data.char_table import CharacterTable
from models.ctc_model import MyCallback
import os
from sources.compiled import CompilationSource
from train_ctc import LabelSource, CtcGenerator
from data.preprocessing import PreProcessor
from config import CTCConfig
from train_ctc import build_model


def model_demo(model, data_path):
    char_table = CharacterTable()

    num_features = CTCConfig().config_dict['num_features']

    train_source = CompilationSource(
        os.path.join(data_path, 'train.h5py'), 10
    )

    print('len', len(train_source))

    test_source = CompilationSource(
        os.path.join(data_path, 'test.h5py'), 10
    )

    train_source = LabelSource(train_source, char_table)
    test_source = LabelSource(test_source, char_table)

    preprocessor = PreProcessor()
    train_gen = CtcGenerator(char_table, train_source, preprocessor, channels=num_features)
    test_gen = CtcGenerator(char_table, test_source, preprocessor, channels=num_features)

    callback = MyCallback(model, train_gen, test_gen, char_table)
    callback.demo(train_gen)
    callback.demo(test_gen)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./compiled')
    parser.add_argument('--cuda', type=bool, default=False)

    args = parser.parse_args()

    ctc_model = build_model(args.cuda, warp=False)

    model_demo(ctc_model.inference_model, args.data_path)
