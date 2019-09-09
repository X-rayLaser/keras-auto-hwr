from sources.compiled import CompilationSource
from data.generators import AutoEncoderGenerator
from models.seq2seq import Seq2seqAutoencoder
from data.preprocessing import PreProcessor
from sources.iam_online import StrokesSource
from sources.preloaded import PreLoadedSource


def padded_source(source, max_len):
    strokes = []
    dummies = []

    for seq_in, dummy in source.get_sequences():
        seq = list(seq_in)
        while len(seq) < max_len:
            seq.append((0, 0))

        seq = seq[:max_len]
        strokes.append(seq)

        dummies.append(dummy)

    return PreLoadedSource(strokes, dummies)


def train(num_train_strokes, num_val_strokes, embedding_size, lrate, epochs, save_path):
    compilation_train_source = CompilationSource('./compiled/train.json')

    compilation_validation_source = CompilationSource('./compiled/validation.json')

    strokes_train_source = StrokesSource(compilation_train_source, num_train_strokes)
    strokes_val_source = StrokesSource(compilation_validation_source, num_val_strokes)

    max_len = max([len(seq_in) for seq_in, _ in strokes_train_source.get_sequences()])
    max_len = max(max_len, max([len(seq_in) for seq_in, _ in strokes_val_source.get_sequences()]))

    print('MAX LEN IS', max_len)

    strokes_train_source = padded_source(strokes_train_source, max_len)
    strokes_val_source = padded_source(strokes_val_source, max_len)

    preprocessor = PreProcessor()

    #train_gen = AutoEncoderGenerator(strokes_train_source,
    #                                 pre_processor=preprocessor, channels=2)

    #val_gen = AutoEncoderGenerator(strokes_val_source,
    #                               pre_processor=preprocessor, channels=2)

    train_gen = FeedForwardAutoEncoderGenerator(strokes_train_source, preprocessor, 2)
    val_gen = FeedForwardAutoEncoderGenerator(strokes_val_source, preprocessor, 2)

    #auto_encoder = Seq2seqAutoencoder(encoding_size=embedding_size, input_channels=2, output_channels=2)
    auto_encoder = VanillaAutoEncoder(max_len * 2, embedding_size)

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


from data.generators import BaseGenerator
from models.base import BaseEncoderDecoder


class VanillaAutoEncoder(BaseEncoderDecoder):
    def __init__(self, seq_len, encoder_size):
        from keras.layers import Input, Dense
        from keras.models import Model

        encoder_input = Input(shape=(seq_len,))
        encoder = Dense(units=encoder_size, activation='tanh')

        decoder = Dense(units=seq_len, activation=None)
        decoder_input = Input(shape=(encoder_size,))

        x = encoder_input
        x = encoder(x)
        self._encoder_model = Model(input=encoder_input, output=x)

        x = decoder_input
        x = decoder(x)
        self._decoder_model = Model(input=decoder_input, output=x)

        x = encoder_input
        x = self._encoder_model(x)
        x = self._decoder_model(x)

        self._auto_encoder = Model(input=encoder_input, output=x)

    def get_encoder(self):
        return self._encoder_model

    def get_decoder(self):
        return self._decoder_model

    def fit_generator(self, lr, train_gen, val_gen, *args, **kwargs):
        predictor = self.get_inference_model()
        from keras.callbacks import Callback
        from util import visualize_stroke

        class MyCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                num_trials = 10
                if epoch % 10 == 0 and epoch > 0:
                    counter = 0
                    for noisy, _ in val_gen.get_examples(batch_size=1):
                        if counter > num_trials:
                            return

                        it = iter(noisy[0].tolist())
                        noisy_points = []

                        for x, y in zip(it, it):
                            noisy_points.append((x, y))

                        noisy_points = noisy_points[1:-1]
                        visualize_stroke(noisy_points).show()
                        x_recovered = predictor.predict(noisy)
                        x_recovered = x_recovered[1:-1]
                        visualize_stroke(x_recovered).show()
                        counter += 1
                        input('Enter any key')
        callbacks = [MyCallback()]
        from keras.optimizers import RMSprop
        self._auto_encoder.compile(optimizer=RMSprop(lr=lr), loss='mean_squared_error',
                                   metrics=['mse'])
        self._auto_encoder.fit_generator(callbacks=callbacks, *args, **kwargs)

    def get_inference_model(self):
        auto_encoder = self._auto_encoder

        class Predictor:
            def predict(self, x):
                v = auto_encoder.predict(x)[0]

                it = iter(v.tolist())
                points = []

                for x, y in zip(it, it):
                    points.append((x, y))
                return points

        return Predictor()


class FeedForwardAutoEncoderGenerator(BaseGenerator):
    def prepare_batch(self, seqs_in, seqs_out):
        import numpy as np
        m = len(seqs_in)
        n = len(seqs_in[0])
        x = np.array(seqs_in)

        x = x.reshape((m, n * 2))
        #x_noisy = x + np.random.normal(loc=0, scale=10, size=x.shape)
        x_noisy = x

        return x_noisy, x


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
