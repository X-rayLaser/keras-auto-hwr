from data import DataFactory, CharacterTable
from sources.compiled import CompilationSource
from sources.iam_online import OnlineSource, LinesSource, WordsSource
from sources.synthetic import SyntheticSource
from models import SequenceToSequenceTrainer
from estimate import CharacterErrorRate


def train(data_path, max_examples, lrate, epochs):
    charset = ''.join([chr(i) for i in range(32, 128)])
    char_table = CharacterTable(charset)

    #source = CompilationSource(data_path)
    source = LinesSource(OnlineSource(data_path))
    #source = SyntheticSource(num_lines=100)

    factory = DataFactory(source, char_table,
                          num_examples=max_examples)

    train_gen = factory.training_generator()
    val_gen = factory.validation_generator()

    trainer = SequenceToSequenceTrainer(char_table=char_table)
    batch_size = 1
    trainer.fit_generator(
        lrate,
        train_gen,
        train_gen.get_examples(batch_size=batch_size),
        steps_per_epoch=int(len(train_gen) / batch_size) + 1,
        validation_data=val_gen.get_examples(batch_size),
        validation_steps=1,
        epochs=epochs)

    estimator = CharacterErrorRate(trainer.get_inference_model(), num_trials=100)
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
    print('Done!')
