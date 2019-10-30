from data.data_set_home import DataSetHome, create_random_source
from train_ctc import build_model
from api import CompilationHome
from data.example_adapters import CTCAdapter
from data.generators import MiniBatchGenerator


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ds1')
    parser.add_argument('--cuda', type=bool, default=False)

    args = parser.parse_args()

    location = CompilationHome(args.dataset_name).root_dir
    home = DataSetHome(location, create_random_source)

    encoding_table = home.get_encoding_table()
    ctc_model = build_model(args.cuda, encoding_table=encoding_table)

    train_source, val_source, test_slice = home.get_slices()

    sentinel = encoding_table.sentinel
    adapter = CTCAdapter(y_padding=sentinel)

    test_gen = MiniBatchGenerator(test_slice, adapter, batch_size=1)

    model = ctc_model.compile_model(0.001)

    res = model.evaluate_generator(test_gen.get_examples(), steps=len(test_slice))
    print('evaluation on test data:', res)
