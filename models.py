import numpy as np
import os

from keras.utils import to_categorical
from keras.models import load_model, Sequential
from keras.models import Model
from keras.layers import SimpleRNN, TimeDistributed, Dense, Input, Conv2D, Flatten, MaxPool2D, BatchNormalization


class SequenceToSequenceTrainer:
    def __init__(self, decimal_table, encoding_size=32):
        self._decimal_table = decimal_table
        self._encoding_size = encoding_size

        self._encoder = self.encoder_model()
        self._decoder = self.decoder_model()

        encoder_inputs = self._encoder.input
        decoder_inputs = self._decoder.input[0]

        state_vector = self._encoder(encoder_inputs)
        output = self._decoder([decoder_inputs, state_vector])

        self._model = Model([encoder_inputs, decoder_inputs], output)

    def encoder_model(self):
        encoder_inputs = Input(shape=(None, 1))

        rnn = SimpleRNN(units=self._encoding_size, return_state=True)

        x = encoder_inputs
        x = BatchNormalization()(x)
        x, encoder_state = rnn(x)

        return Model(encoder_inputs, encoder_state)

    def decoder_model(self):
        decoder_states = self._encoding_size

        decoder_inputs = Input(shape=(None, len(self._decimal_table)))
        initial_state = Input(shape=(self._encoding_size,))

        rnn = SimpleRNN(units=decoder_states,
                        input_shape=(None, self._encoding_size),
                        return_sequences=True,
                        name='decoder_first_layer')
        densor = TimeDistributed(Dense(units=len(self._decimal_table),
                                       activation='softmax'))

        x = decoder_inputs
        x = rnn(x, initial_state=initial_state)
        decoder_output = densor(x)
        return Model([decoder_inputs, initial_state], decoder_output)

    def save(self, path):
        self._encoder.save_weights(os.path.join(path, 'encoder.h5'))
        self._decoder.save_weights(os.path.join(path, 'decoder.h5'))

    def load(self, path):
        self._encoder.load_weights(os.path.join(path, 'encoder.h5'))
        self._decoder.load_weights(os.path.join(path, 'decoder.h5'))

    def fit_generator(self, *args, **kwargs):
        self._model.compile(optimizer='adam', loss='categorical_crossentropy',
                            metrics=['accuracy'])
        self._model.summary()
        self._model.fit_generator(*args, **kwargs)
