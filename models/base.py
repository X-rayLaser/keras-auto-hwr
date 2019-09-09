import numpy as np
import os


class BaseModel:
    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def fit_generator(self, lr, train_gen, val_gen, *args, **kwargs):
        raise NotImplementedError

    def get_inference_model(self):
        raise NotImplementedError

    def get_performance_estimator(self, num_trials):
        raise NotImplementedError


class BaseEncoderDecoder(BaseModel):
    def get_encoder(self):
        raise NotImplementedError

    def get_decoder(self):
        raise NotImplementedError

    def save(self, path):
        self.get_encoder().save_weights(os.path.join(path, 'encoder.h5'))
        self.get_decoder().save_weights(os.path.join(path, 'decoder.h5'))

    def load(self, path):
        self.get_encoder().load_weights(os.path.join(path, 'encoder.h5'))
        self.get_decoder().load_weights(os.path.join(path, 'decoder.h5'))


class BeamCandidate:
    def __init__(self, full_sequence, character, likelihood, state):
        self.full_sequence = full_sequence
        self.character = character
        self.likelihood = likelihood
        self.state = state

    def branch_off(self, character, likelihood, state):
        seq = self.full_sequence + character
        return BeamCandidate(seq, character, likelihood, state)


class BaseBeamSearch:
    def __init__(self, char_table, beam_size=3, max_len=150):
        self._char_table = char_table
        self._beam_size = beam_size
        self._max_len = max_len

    def generate_sequence(self):
        char_table = self._char_table
        ch = char_table.start

        decoder_state = self.get_initial_state()
        candidates = [BeamCandidate(full_sequence=ch, character=ch,
                                    likelihood=0, state=decoder_state)]

        return self._beam_search(candidates)

    def get_initial_state(self):
        raise NotImplementedError

    def decode_next(self, prev_y, prev_state):
        raise NotImplementedError

    def _check_candidate(self, candidate):
        index = self._char_table.encode(candidate.character)
        v = np.zeros((1, 1, len(self._char_table)))
        v[0, 0, index] = 1
        next_p, next_state = self.decode_next(v, candidate.state)

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
