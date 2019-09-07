from sources.compiled import CompilationSource
from data.generators import AutoEncoderGenerator
from models.seq2seq import Seq2seqAutoencoder
from data.preprocessing import PreProcessor
from sources.iam_online import StrokesSource


def train(num_train_strokes, num_val_strokes, embedding_size, lrate, epochs, save_path):
    compilation_train_source = CompilationSource('./compiled/train.json')

    compilation_validation_source = CompilationSource('./compiled/validation.json')

    strokes_train_source = StrokesSource(compilation_train_source, num_train_strokes)
    strokes_val_source = StrokesSource(compilation_validation_source, num_val_strokes)

    preprocessor = PreProcessor()

    train_gen = AutoEncoderGenerator(strokes_train_source,
                                     pre_processor=preprocessor, channels=2)

    val_gen = AutoEncoderGenerator(strokes_val_source,
                                   pre_processor=preprocessor, channels=2)

    auto_encoder = Seq2seqAutoencoder(encoding_size=embedding_size, input_channels=2, output_channels=2)

    batch_size = 1

    auto_encoder.fit_generator(
        lrate,
        train_gen,
        val_gen,
        train_gen.get_examples(batch_size=batch_size),
        steps_per_epoch=int(len(train_gen) / batch_size) + 1,
        validation_data=val_gen.get_examples(batch_size),
        validation_steps=num_val_strokes,
        epochs=epochs)

    auto_encoder.save(save_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_strokes', type=int, default=128)
    parser.add_argument('--val_strokes', type=int, default=128)
    parser.add_argument('--lrate', type=float, default=0.001)
    parser.add_argument('--embedding_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--save_path', type=str)

    args = parser.parse_args()

    train(num_train_strokes=args.train_strokes,
          num_val_strokes=args.val_strokes, embedding_size=args.embedding_size,
          lrate=args.lrate, epochs=args.epochs, save_path=args.save_path)
