from data.char_table import CharacterTable
from sources.compiled import CompilationSource
from models.seq2seq import SequenceToSequenceTrainer
from data.generators import DataSetGenerator
import os
from data.preprocessing import PreProcessor


def train(data_path, max_examples, lrate, epochs):
    char_table = CharacterTable()

    train_path = os.path.join(data_path, 'train.h5py')
    val_path = os.path.join(data_path, 'validation.h5py')

    train_source = CompilationSource(train_path, max_examples)

    val_source = CompilationSource(val_path, max(1, max_examples // 2))

    trainer = SequenceToSequenceTrainer(char_table, input_channels=4)

    preprocessor = PreProcessor()
    train_gen = DataSetGenerator(train_source, char_table, preprocessor, channels=4)
    val_gen = DataSetGenerator(val_source, char_table, preprocessor, channels=4)

    batch_size = 1
    validation_steps = len(val_source)
    trainer.fit_generator(
        lrate,
        train_gen,
        val_gen,
        train_gen.get_examples(batch_size=batch_size),
        steps_per_epoch=len(train_gen) / batch_size,
        validation_data=val_gen.get_examples(batch_size),
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
