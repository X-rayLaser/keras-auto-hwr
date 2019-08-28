from data import CharacterTable, Seq2seqFactory, AttentionalSeq2seqFactory
from sources.compiled import CompilationSource
from sources.iam_online import OnlineSource, LinesSource, WordsSource
from sources.synthetic import SyntheticSource
from estimate import Seq2seqMetric


def train(data_path, max_examples, lrate, epochs):
    charset = ''.join([chr(i) for i in range(32, 128)])
    char_table = CharacterTable(charset)

    source = CompilationSource(data_path)
    #source = LinesSource(OnlineSource(data_path))
    #source = SyntheticSource(num_lines=100)

    factory = AttentionalSeq2seqFactory(Tx=350, Ty=10, num_cells=128,
                                        data_source=source,
                                        char_table=char_table,
                                        num_examples=max_examples)

    factory.prepare_sources()
    train_gen = factory.training_generator()
    val_gen = factory.validation_generator()

    trainer = factory.create_model()
    batch_size = 16
    validation_steps = 4
    trainer.fit_generator(
        lrate,
        train_gen,
        val_gen,
        train_gen.get_examples(batch_size=batch_size),
        steps_per_epoch=int(len(train_gen) / batch_size) + 1,
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
