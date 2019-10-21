from unittest import TestCase
import numpy as np


class TokenPassing:
    def score(self, word_model, path_distribution):
        score = 0

        scores = np.ones_like(path_distribution) * 10**12

        for state in word_model.states:
            states = word_model.outgoing_states(state)

            for st in states:
                code = word_model.get_code(st)
                p = path_distribution[0, code]
                - np.log(p)

        for elem in path_distribution:
            score += word_model.local_cost(0, elem)

        return word_model.transition_cost(0, 0) + word_model.local_cost(0, 0)

    def decode(self, path_distribution):
        return []


class WordModel:
    def __init__(self, codes):
        self._codes = codes

    def outgoing_states(self, state):
        if state == len(self._codes - 1):
            return [state]
        return [state, state + 1]

    @property
    def states(self):
        return list(range(len(self._codes)))

    def get_code(self, state):
        return self._codes[state]


class NonSquaredTransitionMatrixException(Exception):
    pass


class DimensionsMismatchException(Exception):
    pass


class TokenPassingTests(TestCase):
    def setUp(self):
        self.decoder = TokenPassing()

    def test_decode_empty_list(self):
        self.assertEqual([], self.decoder.decode([]))


class State:
    infinite_score = np.inf

    def __init__(self, state, probabilities):
        self.state = state
        self.p = probabilities
        self._transitions = []

        self._score = self.infinite_score
        self._history = []

        self._staging = None
        self._step = 0

    def _flush_staging(self):
        self._staging = None

    def initialize_token(self):
        self._score = 0

    def pass_token(self, token):
        if self._staging is None:
            self._staging = token
            return

        score, history = token
        current_score, history = self._staging
        if score < current_score:
            self._staging = token

    def commit(self):
        if self._staging:
            self._score, self._history = self._staging
            self._flush_staging()

        self._step += 1

    def local_cost(self):
        index = self._step
        return - np.log(self.p[index])

    @property
    def transitions(self):
        return self._transitions

    def add_transition(self, transition):
        self._transitions.append(transition)

    @property
    def token(self):
        return self._score, self._history


class NullState(State):
    def __init__(self):
        super().__init__(0, [])

    def pass_token(self, token):
        pass

    def local_cost(self):
        return self.infinite_score

    @property
    def token(self):
        if self._step == 0:
            return 0, []

        return self.infinite_score, []


class Transition:
    @staticmethod
    def free(destination):
        return Transition(destination, 1.0)

    def __init__(self, destination, probability):
        self.destination = destination
        self.probability = probability

    def cost(self):
        return - np.log(self.probability)


class StateTests(TestCase):
    def setUp(self):
        self.state_value = 25
        self.p = [0.5, 0.2, 0.1]

    def test_local_cost(self):
        state = State(self.state_value, [1])
        self.assertEqual(0, state.local_cost())

        state = State(self.state_value, self.p)
        self.assertEqual(- np.log(self.p[0]), state.local_cost())
        state.commit()
        self.assertEqual(- np.log(self.p[1]), state.local_cost())

    def test_no_initial_transitions(self):
        state = State(self.state_value, self.p)
        self.assertEqual(0, len(state.transitions))

    def test_transition_cost(self):
        state = State(self.state_value, self.p)
        transition = Transition(state, 1)
        self.assertEqual(0, transition.cost())

        transition = Transition(state, 0.2)
        self.assertEqual(- np.log(0.2), transition.cost())

        transition = Transition.free(state)
        self.assertEqual(0, transition.cost())

    def test_add_transition(self):
        state = State(self.state_value, self.p)

        dest = State(2, self.p)
        transition = Transition(dest, 0.2)
        state.add_transition(transition)
        self.assertEqual(1, len(state.transitions))

        destination_state = state.transitions[0].destination
        self.assertEqual(2, destination_state.state)
        self.assertEqual(self.p, destination_state.p)

    def test_initial_token(self):
        state = State(self.state_value, self.p)
        score, history = state.token
        self.assertEqual(state.infinite_score, score)
        self.assertEqual([], history)

    def test_token_after_calling_initialization_method(self):
        state = State(self.state_value, self.p)
        state.initialize_token()
        score, history = state.token
        self.assertEqual(0, score)
        self.assertEqual([], history)

    def test_token_after_passing_token_and_commit(self):
        state = State(self.state_value, self.p)

        token = (45.3, [state])
        state.pass_token(token)
        state.pass_token((20.3, []))

        state.commit()
        score, history = state.token
        self.assertEqual(20.3, score)
        self.assertEqual([], history)

    def test_before_commit(self):
        state = State(self.state_value, self.p)

        token = (45.3, [])
        state.pass_token(token)

        score, history = state.token

        self.assertEqual(state.infinite_score, score)
        self.assertEqual([], history)

    def test_pass_token_after_commit(self):
        state = State(self.state_value, self.p)
        token = (45.3, [state])
        state.pass_token(token)
        state.commit()
        state.pass_token((80.3, []))
        state.pass_token((50.3, []))
        state.pass_token((90.3, []))

        state.commit()

        score, history = state.token
        self.assertEqual(50.3, score)
        self.assertEqual([], history)

    def test_commit_without_passing_token(self):
        state = State(self.state_value, self.p)
        token = (45.3, [state])
        state.pass_token(token)
        state.commit()

        state.commit()

        score, history = state.token
        self.assertEqual(45.3, score)
        self.assertEqual([state], history)


class StatesMachine:
    def __init__(self, states, steps):
        initial_state = NullState()
        self._states = states + [initial_state]

        for state in states:
            transition = Transition.free(state)
            initial_state.add_transition(transition)

    def connect(self, i, j):
        transition = Transition.free(self._states[j])
        self._states[i].add_transition(transition)

    def best_path(self):
        score, history = self._optimal_path()
        return history

    def _optimal_path(self):
        min_score = np.inf
        min_history = self._states[0].token[1]
        for state in self._states:
            score, history = state.token
            if score < min_score:
                min_score = score
                min_history = history

        return min_score, min_history

    def best_score(self):
        score, history = self._optimal_path()
        return score

    def evolve(self):
        for state in self._states:
            for transition in state.transitions:
                cost = transition.cost()
                score, history = state.token
                destination_state = transition.destination

                new_score = score + cost + destination_state.local_cost()
                new_history = history + [destination_state.state]

                transition.destination.pass_token((new_score, new_history))

        for state in self._states:
            state.commit()


class StateMachineTests(TestCase):
    def setUp(self):
        self.states = [State(50, [0.5, 0.1]), State(20, [0.1, 0.2])]

    def test_best_path_initially(self):
        states = [State(50, [0.5, 0.1]), State(20, [0.1, 0.2])]
        machine = StatesMachine(states, 2)
        machine.connect(0, 1)
        machine.connect(1, 1)
        self.assertEqual([], machine.best_path())

    def test_one_state_model(self):
        states = [State(50, [0.5, 0.1])]
        machine = StatesMachine(states, 2)
        machine.connect(0, 0)

        machine.evolve()
        self.assertEqual(- np.log(0.5), machine.best_score())
        self.assertEqual([50], machine.best_path())

        machine.evolve()
        self.assertEqual(- np.log(0.5) - np.log(0.1), machine.best_score())
        self.assertEqual([50, 50], machine.best_path())

    def test_multiple_states_model(self):
        states = [State(50, [0.5, 0.1]), State(20, [0.1, 0.2])]
        machine = StatesMachine(states, 2)
        machine.connect(0, 1)
        machine.connect(1, 1)

        machine.evolve()
        self.assertEqual([50], machine.best_path())
        self.assertEqual(- np.log(0.5), machine.best_score())

        machine.evolve()
        self.assertEqual([50, 20], machine.best_path())
        self.assertEqual(- np.log(0.5) - np.log(0.2), machine.best_score())


# todo: composite pattern: SentenceModel.passToken(), WordModel.passToken(src, dest), Node.passToken(src, dest)
"""
def iteration(self)
    for state in states:
        out_states = model.outgoing(state)
        for out_state in out_states:
            model.passToken(state, out_state)
"""
