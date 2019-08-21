import numpy as np
import os

from keras.utils import to_categorical
from keras.models import load_model, Sequential
from keras.models import Model
from keras.layers import SimpleRNN, TimeDistributed, Dense, Input, Conv2D, Flatten, MaxPool2D, BatchNormalization, GRU


class SequenceToSequenceTrainer:
    def __init__(self, char_table, encoding_size=16):
        self._char_table = char_table
        self._encoding_size = encoding_size

        self._encoder = self.encoder_model()
        self._decoder = self.decoder_model()

        encoder_inputs = self._encoder.input
        decoder_inputs = self._decoder.input[0]

        state_vector = self._encoder(encoder_inputs)
        output, _ = self._decoder([decoder_inputs, state_vector])

        self._model = Model([encoder_inputs, decoder_inputs], output)

    def encoder_model(self):
        encoder_inputs = Input(shape=(None, 1))

        rnn = GRU(units=self._encoding_size, return_state=True)

        x = encoder_inputs
        x = BatchNormalization()(x)
        x, encoder_state = rnn(x)

        return Model(encoder_inputs, encoder_state)

    def decoder_model(self):
        decoder_states = self._encoding_size

        decoder_inputs = Input(shape=(None, len(self._char_table)))
        initial_state = Input(shape=(self._encoding_size,))

        rnn = SimpleRNN(units=decoder_states,
                        input_shape=(None, self._encoding_size),
                        return_sequences=True,
                        return_state=True,
                        name='decoder_first_layer')
        densor = TimeDistributed(Dense(units=len(self._char_table),
                                       activation='softmax'))

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

    def fit_generator(self, *args, **kwargs):
        from keras.optimizers import Adam
        self._model.compile(optimizer='adam', loss='categorical_crossentropy',
                            metrics=['accuracy'])
        self._model.summary()
        self._model.fit_generator(*args, **kwargs)

    def get_inference_model(self):
        return ImageToSequencePredictor(self._encoder, self._decoder,
                                        self._char_table)


class ImageToSequencePredictor:
    def __init__(self, encoder, decoder, char_table):
        self._encoder = encoder
        self._decoder = decoder
        self._char_table = char_table

    @property
    def char_table(self):
        return self._char_table

    def predict(self, hand_writing):
        hand_writing = hand_writing.reshape(-1, hand_writing.shape[1], 1)
        state = self._encoder.predict(hand_writing)

        char_table = self._char_table
        ch = char_table.start

        digitstr = ch

        count = 0
        while ch != char_table.sentinel:
            ch = digitstr[-1]
            index = char_table.encode(ch)

            v = np.zeros((1, 1, len(char_table)))
            v[0, 0, index] = 1

            probs, state = self._decoder.predict([v, state])

            cls = np.argmax(probs[0][-1])
            ch = char_table.decode(cls)
            digitstr += ch

            count += 1
            if count >= 150:
                break

        return digitstr[1:]
