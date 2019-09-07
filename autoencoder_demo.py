from util import visualize_stroke
from models.seq2seq import Seq2seqAutoencoder
from sources.compiled import CompilationSource
from sources.iam_online import StrokesSource
from data.generators import AutoEncoderGenerator
from data.preprocessing import PreProcessor


def generate(num_trials=10, embedding_size=16):
    auto_encoder = Seq2seqAutoencoder(encoding_size=embedding_size, input_channels=2,
                                      output_channels=2)
    auto_encoder.load('./weights/auto_encoder')

    predictor = auto_encoder.get_inference_model()

    source = CompilationSource(path='./compiled/validation.json')
    stroke_source = StrokesSource(source, num_strokes=num_trials)
    preprocessor = PreProcessor()
    val_gen = AutoEncoderGenerator(strokes_iterator=stroke_source,
                                   pre_processor=preprocessor, channels=2)

    counter = 0
    for [x_noisy, y_in], y_out in val_gen.get_examples(batch_size=1):
        if counter > num_trials:
            break

        noisy_points = []
        for v in x_noisy[0]:
            noisy_points.append((v[0], v[1]))

        noisy_points = noisy_points[1:-1]
        x_recovered = predictor.predict(x_noisy)
        x_recovered = x_recovered[1:-1]
        yield noisy_points, x_recovered
        counter += 1


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trials', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--save_path', type=str, default='')

    args = parser.parse_args()

    noisy_dir = os.path.join(args.save_path, 'noisy')
    recovered_dir = os.path.join(args.save_path, 'recovered')
    for i, (noisy, recovered) in enumerate(generate(args.num_trials,
                                                    args.embedding_size)):
        noisy_image = visualize_stroke(noisy)
        recovered_image = visualize_stroke(recovered)
        if args.save_path:
            os.makedirs(noisy_dir, exist_ok=True)
            os.makedirs(recovered_dir, exist_ok=True)
            noisy_path = os.path.join(noisy_dir, '{}.jpg'.format(i))
            recovered_path = os.path.join(recovered_dir, '{}.jpg'.format(i))

            noisy_image.save(noisy_path)
            recovered_image.save(recovered_path)
        else:
            noisy_image.show()
            recovered_image.show()

            input('Enter any key')
