from unittest import TestCase
from models.base import BaseBeamSearch, BeamCandidate
from data.char_table import CharacterTable
from models.base import PathBuilder


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
            code = char_table.encode(char_table.sentinel)
        else:
            ch = self.result[self.counter]
            code = char_table.encode(ch)

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


class PathBuilderTests(TestCase):
    def test_after_initialization(self):
        roots = [(2, 0.1, 0.1), (5, 0.3, 0.3)]
        builder = PathBuilder(roots)
        self.assertEqual(builder.best_path, [5])

        self.assertEqual(len(builder.paths), 2)
        self.assertEqual(builder.paths[0], [2])
        self.assertEqual(builder.paths[1], [5])

    def test_make_step(self):
        roots = [(0, 0.1, 0.1)]
        builder = PathBuilder(roots)

        pmfs = [
            [0.3, 0.5, 0.2]
        ]
        builder.make_step(pmfs)

        self.assertEqual(builder.best_path, [0, 1])


class BeamSearchTests(TestCase):
    def test_max_length_constraint(self):
        char_table = CharacterTable()
        max_len = 5

        decoder = DummyDecoder()

        search = DummyBeamSearch(decoder, char_table, beam_size=3,
                                 max_len=max_len)

        s = search.generate_sequence()
        self.assertEqual(len(s), max_len)

    def test_stop_generating_after_end_of_sequence_code(self):
        char_table = CharacterTable()

        result = 'hi' + char_table.sentinel
        decoder = PredeterminedDecoder(result)

        search = DummyBeamSearch(decoder, char_table)

        s = search.generate_sequence()

        self.assertEqual(len(s), len(result))

    def test_greedy_search(self):
        char_table = CharacterTable()

        result = 'Hello, world!' + char_table.sentinel
        decoder = PredeterminedDecoder(result)

        search = DummyBeamSearch(decoder, char_table, beam_size=1)

        s = search.generate_sequence()

        self.assertEqual(s, result)


# todo: finish development of PathBuilder class
# todo: change implementation of BaseBeamSearch
# todo: test beam search with beam_size > 1
