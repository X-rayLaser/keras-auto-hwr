from unittest import TestCase
from data.encodings import CharacterTable
from algorithms.beam_search import SearchPath, PathBuilder, WrongNumberOfPMFsException, StatesKeeper, BaseBeamSearch, \
    BeamCandidate
import numpy as np


class DummyDecoder:
    def get_initial_state(self):
        char_table = CharacterTable()
        epsilon = 0.001
        return [epsilon] * len(char_table)

    def decode_next(self, prev_y, prev_state):
        char_table = CharacterTable()
        epsilon = 0.001
        next_p = [epsilon] * len(char_table)
        next_state = [epsilon] * len(char_table)
        return next_p, next_state


class PredeterminedDecoder(DummyDecoder):
    def __init__(self, result):
        self.counter = 0
        self.result = result

    def decode_next(self, prev_y, prev_state):
        char_table = CharacterTable()
        epsilon = 0.001
        next_p = [epsilon] * len(char_table)
        next_state = [epsilon] * len(char_table)

        if self.counter >= len(self.result):
            code = 27
        else:
            code = self.result[self.counter]

        next_p[code] = 1.0

        self.counter += 1
        return next_p, next_state


class DummyBeamSearch(BaseBeamSearch):
    def __init__(self, decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = decoder

    def get_initial_state(self):
        return self.decoder.get_initial_state()

    def decode_next(self, prev_y, prev_state):
        return self.decoder.decode_next(prev_y, prev_state)


class BeamCandidateTests(TestCase):
    def test_branch_off(self):
        candidate = BeamCandidate(full_sequence='Hello', character='o',
                                  likelihood=0.3, state=9)

        self.assertEqual(candidate.full_sequence, 'Hello')

        self.assertEqual(candidate.character, 'o')
        self.assertEqual(candidate.likelihood, 0.3)
        self.assertEqual(candidate.state, 9)

        next_candidate = candidate.branch_off(character=',',
                                              likelihood=0.5, state=4)

        self.assertEqual(next_candidate.full_sequence, 'Hello,')
        self.assertEqual(next_candidate.character, ',')
        self.assertEqual(next_candidate.likelihood, 0.5)
        self.assertEqual(next_candidate.state, 4)


class PathTests(TestCase):
    def setUp(self):
        self.labels = [3, 0]
        self.probs = [0.2, 0.5]

    def test_appended(self):
        label1, label2 = self.labels
        p1, p2 = self.probs
        path = SearchPath()
        path = path.branch_off(label=label1, p=p1)

        self.assertEqual(path.labels, [label1])

        path = path.branch_off(label=label2, p=p2)
        self.assertEqual(path.labels, self.labels)

    def test_likelihood_is_zero_initially(self):
        path = SearchPath()
        self.assertEqual(path.likelihood, 0)

    def test_likelihood(self):
        path = SearchPath()
        label1, label2 = self.labels
        p1, p2 = self.probs

        path = path.branch_off(label1, p1)
        self.assertEqual(path.likelihood, p1)

        path = path.branch_off(label2, p2)
        self.assertEqual(path.likelihood, p1 * p2)

    def test_no_side_effect(self):
        path = SearchPath()
        label1, label2 = self.labels
        p1, p2 = self.probs

        path_before = path.branch_off(label1, p1)
        path_before.branch_off(label2, p2)
        self.assertEqual(path.labels, [])
        self.assertEqual(path_before.labels, [label1])
        self.assertEqual(path_before.likelihood, p1)


class PathBuilderTests(TestCase):
    def test_after_initialization(self):
        label1 = 2
        label2 = 5
        roots = [(label1, 0.1), (label2, 0.3)]
        expected_paths = [[label1], [label2]]

        builder = PathBuilder(roots)
        self.assertEqual(builder.best_path, expected_paths[1])

        self.assertEqual(len(builder.paths), 2)
        self.assertEqual(builder.paths[0], [label1])
        self.assertEqual(builder.paths[1], [label2])

    def test_make_step(self):
        roots = [(0, 0.1)]
        builder = PathBuilder(roots)

        pmfs = [
            [0.3, 0.5, 0.2]
        ]
        builder.make_step(pmfs)

        self.assertEqual(builder.best_path, [0, 1])

    def test_make_step_throws_exception(self):
        roots = [(0, 0.1)]
        builder = PathBuilder(roots)

        n = 10
        pmfs = [np.zeros(n), np.zeros(n)]
        self.assertRaises(WrongNumberOfPMFsException,
                          lambda: builder.make_step(pmfs))

    def test_with_multiple_paths(self):
        label1 = 0
        label2 = 2

        roots = [(label1, 0.1), (label2, 0.2)]
        builder = PathBuilder(roots)

        pmfs = [
            [0.1, 0.9, 0],
            [0.3, 0.3, 0.4]
        ]
        builder.make_step(pmfs)

        expected_best = [label1, np.argmax(pmfs[0])]

        self.assertEqual(len(builder.paths), len(pmfs))
        self.assertEqual(builder.paths[0], [label1, 1])
        self.assertEqual(builder.paths[1], [label2, 2])
        self.assertEqual(builder.best_path, expected_best)


class StatesKeeperTests(TestCase):
    def test_retrieve_state_for_sequence_of_steps(self):
        s = 5
        keeper = StatesKeeper(initial_state=s)

        self.assertEqual(keeper.retrieve([]), s)

        path = [2, 5, 1, 9]
        new_state = 23
        keeper.store(path=path, state=new_state)
        self.assertEqual(keeper.retrieve(path), new_state)


class BeamSearchTests(TestCase):
    def setUp(self):
        self.result = [3, 12, 44]
        self.sos = 26
        self.eos = 27
        self.beam_size = 1

    def test_max_length_constraint(self):
        max_len = 5

        decoder = DummyDecoder()

        search = DummyBeamSearch(decoder, start_of_seq=self.sos,
                                 end_of_seq=self.eos, beam_size=3,
                                 max_len=max_len)

        s = search.generate_sequence()
        self.assertEqual(len(s), max_len)

    def test_stop_generating_after_end_of_sequence_code(self):
        decoder = PredeterminedDecoder(self.result)

        search = DummyBeamSearch(decoder, start_of_seq=self.sos,
                                 end_of_seq=self.eos, beam_size=self.beam_size)

        s = search.generate_sequence()

        self.assertEqual(len(s), len(self.result))

    def test_greedy_search(self):
        decoder = PredeterminedDecoder(self.result)

        search = DummyBeamSearch(decoder, start_of_seq=self.sos,
                                 end_of_seq=self.eos, beam_size=self.beam_size)

        s = search.generate_sequence()

        self.assertEqual(s, self.result)


# todo: test more edge cases
