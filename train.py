from data.char_table import CharacterTable
from sources.compiled import CompilationSource
from models.seq2seq import SequenceToSequenceTrainer
import os
from data.generators import MiniBatchGenerator
from data.data_set_home import DataSetHome
from api import CompilationHome, create_source
from data.example_adapters import Seq2seqAdapter


def train(data_path, max_examples, lrate, epochs):
    char_table = CharacterTable()

    trainer = SequenceToSequenceTrainer(char_table, input_channels=4)

    location = CompilationHome('ds1').root_dir
    ds_home = DataSetHome(location, create_source)

    train_source, val_source, test_slice = ds_home.get_slices()

    adapter = Seq2seqAdapter()

    train_gen = MiniBatchGenerator(train_source, adapter, batch_size=1)
    val_gen = MiniBatchGenerator(val_source, adapter, batch_size=1)

    batch_size = 1
    validation_steps = len(val_source)
    trainer.fit_generator(
        lrate,
        train_gen,
        val_gen,
        train_gen.get_examples(),
        steps_per_epoch=len(train_gen) / batch_size,
        validation_data=val_gen.get_examples(),
        validation_steps=validation_steps,
        epochs=epochs)

    estimator = trainer.get_performance_estimator(validation_steps)
    error_rate = estimator.estimate(train_gen)

    print(error_rate)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./')
    parser.add_argument('--max_examples', type=int, default=8)
    parser.add_argument('--lrate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()

    train(args.data_path, args.max_examples, args.lrate, args.epochs)
