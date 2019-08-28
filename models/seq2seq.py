import os

import numpy as np
from keras import Model, Input
from keras.layers import SimpleRNN, Bidirectional, Conv1D, MaxPool1D, Reshape, Concatenate, TimeDistributed, Dense

from models import BaseModel
from estimate import CharacterErrorRate
from keras.callbacks import Callback
from keras.optimizers import RMSprop


class SequenceToSequenceTrainer(BaseModel):
    def __init__(self, char_table, encoding_size=128):
        self._char_table = char_table
        self._encoding_size = encoding_size

        self._encoder = self.encoder_model()
        self._before = self._encoder.get_weights()
        self._decoder = self.decoder_model()

        encoder_inputs = self._encoder.input
        decoder_inputs = self._decoder.input[0]

        state_vector = self._encoder(encoder_inputs)
        output, _ = self._decoder([decoder_inputs, state_vector])

        self._model = Model([encoder_inputs, decoder_inputs], output)

    def encoder_model(self):
        encoder_inputs = Input(shape=(None, 1))

        rnn = SimpleRNN(units=self._encoding_size // 2, return_state=True)
        rnn = Bidirectional(rnn)

        x = encoder_inputs
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

        states_seq, forward_state, backward_state = rnn(x)
        encoder_state = Concatenate()([forward_state, backward_state])

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

    def fit_generator(self, lr, train_gen, val_gen, *args, **kwargs):
        estimator = CharacterErrorRate(self.get_inference_model(), 8)

        class MyCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 5 == 0:
                    estimator.estimate(train_gen)
                    print()
                    estimator.estimate(val_gen)

        self._model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy',
                            metrics=['accuracy'])
        self._model.summary()
        self._model.fit_generator(callbacks=[MyCallback()], *args, **kwargs)
        for i in range(len(self._before)):
            dif = self._before[i] - self._encoder.get_weights()[i]
            print('difference', dif.mean(), dif.std())

    def get_inference_model(self):
        return SequenceToSequencePredictor(self._encoder, self._decoder,
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


class BeamSearch:
    def __init__(self, initial_state, char_table, decoder, beam_size=3, max_len=150):
        self._initial_state = initial_state
        self._char_table = char_table
        self._decoder = decoder
        self._beam_size = beam_size
        self._max_len = max_len

    def generate_sequence(self):
        char_table = self._char_table
        ch = char_table.start

        candidates = [BeamCandidate(full_sequence=ch, character=ch,
                                    likelihood=0, state=self._initial_state)]

        return self._beam_search(candidates)

    def _check_candidate(self, candidate):
        index = self._char_table.encode(candidate.character)
        v = np.zeros((1, 1, len(self._char_table)))
        v[0, 0, index] = 1
        prob, next_state = self._decoder.predict([v, candidate.state])

        next_p = prob[0][-1]

        joint_pmf = np.log(next_p) + candidate.likelihood

        return joint_pmf, next_state

    def _next_candidates(self, candidates, joint_pmfs, states):
        a = np.array(joint_pmfs)

        for _ in range(self._beam_size):
            candidate_index = np.argmax(np.max(a, axis=1)).squeeze()
            class_index = np.argmax(a[candidate_index])
            max_p = np.max(a)
            a[candidate_index, class_index] = 0

            char = self._char_table.decode(class_index)
            candidate = candidates[candidate_index]

            yield candidate.branch_off(char, max_p, states[candidate_index])

    def _end_of_sequence(self, best_candidate):
        return best_candidate.character == self._char_table.sentinel or \
               len(best_candidate.full_sequence) > self._max_len

    def _beam_search(self, candidates):
        joint_pmfs = []
        states = []

        for candidate in candidates:
            joint_pmf, next_state = self._check_candidate(candidate)
            joint_pmfs.append(joint_pmf)
            states.append(next_state)

        next_candidates = []
        for next_one in self._next_candidates(candidates, joint_pmfs, states):
            next_candidates.append(next_one)

        best_one = next_candidates[0]
        if self._end_of_sequence(best_one):
            return best_one.full_sequence[1:]

        return self._beam_search(next_candidates)


class SequenceToSequencePredictor:
    def __init__(self, encoder, decoder, char_table):
        self._encoder = encoder
        self._decoder = decoder
        self._char_table = char_table

        self._prev_state = None
        self._prev_hwr = None

    @property
    def char_table(self):
        return self._char_table

    def predict(self, hand_writing):
        hand_writing = hand_writing.reshape(1, hand_writing.shape[0], 1)
        state = self._encoder.predict(hand_writing)

        self._prev_hwr = hand_writing
        self._prev_state = state

        beam_search = BeamSearch(initial_state=state,
                                 char_table=self._char_table,
                                 decoder=self._decoder)
        return beam_search.generate_sequence()