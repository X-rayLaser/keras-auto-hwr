from util import visualize_stroke
from sources.compiled import CompilationSource
from sources.iam_online import StrokesSource
from data.preprocessing import PreProcessor
from train_autoencoder import FeedForwardAutoEncoderGenerator, VanillaAutoEncoder, padded_source


def generate(num_trials=10, embedding_size=16, val_gen=True):
    if val_gen:
        path = './compiled/validation.h5py'
    else:
        path = './compiled/train.h5py'

    source = CompilationSource(path=path, num_lines=num_trials)
    stroke_source = StrokesSource(source, num_strokes=num_trials)

    max_len = 508
    stroke_source = padded_source(stroke_source, max_len)

    preprocessor = PreProcessor()
    gen = FeedForwardAutoEncoderGenerator(stroke_source, preprocessor, channels=2)
    auto_encoder = VanillaAutoEncoder(max_len * 2, embedding_size)
    auto_encoder.load('./weights/auto_encoder')

    predictor = auto_encoder.get_inference_model()

    counter = 0
    for noisy, _ in gen.get_examples(batch_size=1):
        if counter > num_trials:
            return

        it = iter(noisy[0].tolist())
        noisy_points = []

        for x, y in zip(it, it):
            noisy_points.append((x, y))

        noisy_points = noisy_points[1:-1]
        x_recovered = predictor.predict(noisy)
        x_recovered = x_recovered[1:-1]
        yield noisy_points, x_recovered

        counter += 1


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trials', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=16)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--val_gen', type=bool, default=True)

    args = parser.parse_args()

    noisy_dir = os.path.join(args.save_path, 'noisy')
    recovered_dir = os.path.join(args.save_path, 'recovered')
    for i, (noisy, recovered) in enumerate(generate(args.num_trials,
                                                    args.embedding_size,
                                                    True)):
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
