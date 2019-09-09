import os

from keras import Model, Input
from keras.layers import SimpleRNN, Bidirectional, Conv1D, MaxPool1D, Reshape, Concatenate, TimeDistributed, Dense, GRU, CuDNNGRU

from models import BaseModel
from estimate import Seq2seqMetric
from keras.callbacks import Callback
from keras.optimizers import RMSprop
from models.base import BaseBeamSearch
from util import visualize_stroke


class BaseSeq2seq(BaseModel):
    def __init__(self, encoding_size=128, input_channels=1, output_channels=1):
        self._encoding_size = encoding_size
        self._input_channels = input_channels
        self._output_channels = output_channels

        self._encoder = self.encoder_model()
        self._decoder = self.decoder_model()

        encoder_inputs = self._encoder.input
        decoder_inputs = self._decoder.input[0]

        state_vector = self._encoder(encoder_inputs)
        output, _ = self._decoder([decoder_inputs, state_vector])

        self._model = Model([encoder_inputs, decoder_inputs], output)

    def feature_extractor(self, x):
        x = Conv1D(filters=6, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPool1D(pool_size=2)(x)
        x = Conv1D(filters=12, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPool1D(pool_size=2)(x)
        x = Conv1D(filters=24, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPool1D(pool_size=2)(x)
        x = Conv1D(filters=24, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPool1D(pool_size=2)(x)

        x = Conv1D(filters=1, kernel_size=1, activation='relu')(x)
        x = Reshape(target_shape=(-1, 1))(x)

        return x

    def encoder_model(self):
        encoder_inputs = Input(shape=(None, self._input_channels))

        rnn = GRU(units=self._encoding_size // 2, return_state=True)
        rnn = Bidirectional(rnn)

        x = encoder_inputs

        x = self.feature_extractor(x)

        states_seq, forward_state, backward_state = rnn(x)
        encoder_state = Concatenate()([forward_state, backward_state])

        return Model(encoder_inputs, encoder_state)

    def decoder_activation(self):
        return 'softmax'

    def decoder_model(self):
        decoder_states = self._encoding_size

        decoder_inputs = Input(shape=(None, self._output_channels))
        initial_state = Input(shape=(self._encoding_size,))

        rnn = GRU(units=decoder_states,
                        input_shape=(None, self._encoding_size),
                        return_sequences=True,
                        return_state=True,
                        name='decoder_first_layer')

        densor = TimeDistributed(Dense(units=self._output_channels,
                                       activation=self.decoder_activation()))

        x = decoder_inputs

        x, state = rnn(x, initial_state=initial_state)
        decoder_output = densor(x)
        return Model([decoder_inputs, initial_state], [decoder_output, state])

    def save(self, path):
        self._encoder.save_weights(os.path.join(path, 'encoder.h5'))
        self._decoder.save_weights(os.path.join(path, 'decoder.h5'))

    def load(self, path):
        self._encoder.load_weights(os.path.join(path, 'encoder.h5'))
        self._decoder.load_weights(os.path.join(path, 'decoder.h5'))


class Seq2seqAutoencoder(BaseSeq2seq):
    def feature_extractor(self, x):
        return x

    def decoder_activation(self):
        return None

    def get_encoder(self):
        return self._encoder

    def fit_generator(self, lr, train_gen, val_gen, *args, **kwargs):
        predictor = self.get_inference_model()

        class MyCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                num_trials = 10
                if epoch % 2 == 0 and epoch > 0:
                    counter = 0
                    for [x_noisy, y_in], y_out in val_gen.get_examples(batch_size=1):
                        if counter > num_trials:
                            return

                        noisy_points = []
                        for v in x_noisy[0]:
                            noisy_points.append((v[0], v[1]))

                        noisy_points = noisy_points[1:-1]
                        visualize_stroke(noisy_points).show()
                        x_recovered = predictor.predict(x_noisy)
                        x_recovered = x_recovered[1:-1]
                        visualize_stroke(x_recovered).show()
                        counter += 1
                        input('Enter any key')

        from keras.optimizers import Adam
        self._model.compile(optimizer=RMSprop(lr=lr), loss='mean_squared_error',
                            metrics=['mse'])

        callbacks = []
        self._model.fit_generator(callbacks=callbacks, *args, **kwargs)

    def get_inference_model(self):
        return AutoEncoderPredictor(self._encoder, self._decoder, channels=2)


class SequenceToSequenceTrainer(BaseSeq2seq):
    def __init__(self, char_table, encoding_size=128, input_channels=2):
        super().__init__(encoding_size, input_channels=input_channels,
                         output_channels=len(char_table))
        self._char_table = char_table

        self._encoder = self.encoder_model()
        self._decoder = self.decoder_model()

        encoder_inputs = self._encoder.input
        decoder_inputs = self._decoder.input[0]

        state_vector = self._encoder(encoder_inputs)
        output, _ = self._decoder([decoder_inputs, state_vector])

        self._model = Model([encoder_inputs, decoder_inputs], output)

    def feature_extractor(self, x):
        return x

    def fit_generator(self, lr, train_gen, val_gen, *args, **kwargs):
        estimator = self.get_performance_estimator(8)

        class MyCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 5 == 0:
                    estimator.estimate(train_gen)
                    print()
                    estimator.estimate(val_gen)

        self._model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy',
                            metrics=['accuracy'])

        self._model.fit_generator(callbacks=[MyCallback()], *args, **kwargs)

    def get_inference_model(self):
        return SequenceToSequencePredictor(self._encoder, self._decoder,
                                           self._char_table, self._input_channels)

    def get_performance_estimator(self, num_trials):
        return Seq2seqMetric(self.get_inference_model(), num_trials)


class Seq2SeqSearch(BaseBeamSearch):
    def __init__(self, decoder, initial_state, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._decoder = decoder
        self._initial_state = initial_state

    def decode_next(self, prev_y, prev_state):
        prob, next_state = self._decoder.predict([prev_y, prev_state])

        next_p = prob[0][-1]
        return next_p, next_state

    def get_initial_state(self):
        return self._initial_state


class SequenceToSequencePredictor:
    def __init__(self, encoder, decoder, char_table, channels):
        self._encoder = encoder
        self._decoder = decoder
        self._char_table = char_table
        self._channels = channels

    @property
    def char_table(self):
        return self._char_table

    def predict(self, hand_writing):
        hand_writing = hand_writing.reshape(1, hand_writing.shape[0],
                                            self._channels)
        state = self._encoder.predict(hand_writing)

        beam_search = Seq2SeqSearch(initial_state=state,
                                    char_table=self._char_table,
                                    decoder=self._decoder)
        return beam_search.generate_sequence()


class AutoEncoderPredictor:
    def __init__(self, encoder, decoder, channels):
        self._encoder = encoder
        self._decoder = decoder
        self._channels = channels

    def predict(self, x):
        import numpy as np
        prev_state = self._encoder.predict(x)
        points = []
        prev_y = np.zeros(2).reshape((1, 1, self._channels))

        while True:
            prev_y, prev_state = self._decoder.predict([prev_y, prev_state])
            p = prev_y[0][-1]

            points.append(p.tolist())
            if len(points) > 100:
                break

        return points
