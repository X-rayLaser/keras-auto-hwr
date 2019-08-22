import numpy as np
import os

from keras.utils import to_categorical
from keras.models import load_model, Sequential
from keras.models import Model
from keras.layers import SimpleRNN, TimeDistributed, Dense, Input, Conv1D, Flatten, MaxPool1D, BatchNormalization, GRU, Reshape


class SequenceToSequenceTrainer:
    def __init__(self, char_table, encoding_size=128):
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

        rnn = SimpleRNN(units=self._encoding_size, return_state=True)

        x = encoder_inputs
        #x = BatchNormalization()(x)
        x = Conv1D(filters=6, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv1D(filters=12, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPool1D()(x)
        x = Conv1D(filters=24, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPool1D()(x)
        x = Conv1D(filters=1, kernel_size=1, activation='relu')(x)
        #x = Flatten()(x)
        x = Reshape(target_shape=(-1, 1))(x)

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


class BeamCandidate:
    def __init__(self, full_sequence, character, likelihood, state):
        self.full_sequence = full_sequence
        self.character = character
        self.likelihood = likelihood
        self.state = state

    def branch_off(self, character, likelihood, state):
        seq = self.full_sequence + character
        return BeamCandidate(seq, character, likelihood, state)


def beam_search(candidates, decoder, char_table, beam=3, max_seq=150):
    pmfs = []
    states = []

    for candidate in candidates:
        index = char_table.encode(candidate.character)
        v = np.zeros((1, 1, len(char_table)))
        v[0, 0, index] = 1
        prob, next_state = decoder.predict([v, candidate.state])

        next_p = prob[0][-1]
        pmfs.append(next_p * candidate.likelihood)
        states.append(next_state)

    a = np.array(pmfs)

    next_candidates = []

    while len(next_candidates) < beam:
        candidate_index = np.argmax(np.max(a, axis=1)).squeeze()
        class_index = np.argmax(a[candidate_index])
        max_p = np.max(a)
        a[candidate_index, class_index] = 0

        char = char_table.decode(class_index)
        candidate = candidates[candidate_index]

        new_candidate = candidate.branch_off(char, max_p, states[candidate_index])
        next_candidates.append(new_candidate)

        if new_candidate.character == char_table.sentinel or len(new_candidate.full_sequence) > max_seq:
            return new_candidate.full_sequence[1:]

    return beam_search(next_candidates, decoder, char_table)


class ImageToSequencePredictor:
    def __init__(self, encoder, decoder, char_table):
        self._encoder = encoder
        self._decoder = decoder
        self._char_table = char_table

    @property
    def char_table(self):
        return self._char_table

    def predict(self, hand_writing):
        hand_writing = hand_writing.reshape(1, hand_writing.shape[0], 1)
        state = self._encoder.predict(hand_writing)

        char_table = self._char_table
        ch = char_table.start

        candidates = [BeamCandidate(full_sequence=ch, character=ch,
                                    likelihood=1, state=state)]
        return beam_search(candidates, decoder=self._decoder, char_table=self._char_table)
