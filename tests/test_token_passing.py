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


class Node:
    def evolve(self):
        pass

    def add_transition(self, transition):
        raise NotImplementedError

    def optimal_path(self):
        raise NotImplementedError

    @property
    def node_input(self):
        raise NotImplementedError

    @property
    def node_output(self):
        raise NotImplementedError

    @property
    def transitions(self):
        raise NotImplementedError

    def commit(self):
        raise NotImplementedError

    def pass_token(self, token):
        raise NotImplementedError


class State(Node):
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

    def optimal_path(self):
        return self.token


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
    def free(source, destination):
        return Transition(source, destination, 1.0)

    def __init__(self, source, destination, probability):
        self.source = source
        self.destination = destination
        self.probability = probability

    def cost(self):
        return - np.log(self.probability)

    def full_cost(self):
        return self.cost() + self.destination.local_cost()

    def pass_token(self):
        score = self.full_cost()
        old_score, history = self.source.token
        new_score = old_score + score
        new_history = history + [self.destination.state]
        self.destination.pass_token((new_score, new_history))


class TransitionTests(TestCase):
    def setUp(self):
        self.p = [0.5, 0.2, 0.1]
        self.state = State(25, self.p)
        self.p_transit = 0.25
        self.transition = Transition(self.state, self.state, self.p_transit)

    def test_cost(self):
        transition = Transition(self.state, self.state, 1)
        self.assertEqual(0, transition.cost())

        self.assertEqual(- np.log(self.p_transit), self.transition.cost())

        transition = Transition.free(self.state, self.state)
        self.assertEqual(0, transition.cost())

    def test_full_cost(self):
        self.assertEqual(- np.log(0.5) - np.log(self.p_transit),
                         self.transition.full_cost())

    def test_pass_token(self):
        expected_score = self.transition.full_cost()
        self.state.initialize_token()
        self.transition.pass_token()
        self.state.commit()

        score, history = self.state.token
        self.assertEqual(expected_score, score)
        self.assertEqual([self.state.state], history)


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

    def test_add_transition(self):
        state = State(self.state_value, self.p)

        dest = State(2, self.p)
        transition = Transition(state, dest, 0.2)
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


class CompositeNode(Node):
    def evolve(self):
        for node in self.nodes:
            node.evolve()

            for transition in node.transitions:
                transition.pass_token()

        for node in self.nodes:
            node.commit()


class Graph(Node):
    def __init__(self, nodes):
        initial_state = NullState()
        self._nodes = nodes + [initial_state]

        for node in nodes:
            transition = Transition.free(initial_state, node)
            initial_state.add_transition(transition)

    def transitions(self):
        return []

    def commit(self):
        pass

    def add_transition(self, transition):
        return
        self._nodes[i].add_transition(transition)

    def optimal_path(self):
        min_score = np.inf
        min_history = self._nodes[0].token[1]
        for state in self._nodes:
            score, history = state.optimal_path()
            if score < min_score:
                min_score = score
                min_history = history

        return min_score, min_history

    def evolve(self):
        for node in self._nodes:
            node.evolve()

            for transition in node.transitions:
                transition.pass_token()

        for node in self._nodes:
            node.commit()


class StateMachineTests(TestCase):
    def setUp(self):
        self.state1 = 50
        self.state2 = 20

        self.p1 = [0.5, 0.1]
        self.p2 = [0.1, 0.2]
        self.states = [State(self.state1, self.p1), State(self.state2, self.p2)]

    def test_best_path_initially(self):
        machine = Graph(self.states)
        self.states[0].add_transition(
            Transition.free(self.states[0], self.states[1])
        )
        self.states[1].add_transition(
            Transition.free(self.states[1], self.states[1])
        )

        score, history = machine.optimal_path()
        self.assertEqual([], history)

    def test_one_state_model(self):
        states = [self.states[0]]
        machine = Graph(states)
        self.states[0].add_transition(
            Transition.free(self.states[0], self.states[0])
        )

        machine.evolve()
        machine.commit()

        t1_prob = self.p1[0]
        t2_prob = self.p1[1]
        score, history = machine.optimal_path()

        self.assertEqual(- np.log(t1_prob), score)
        self.assertEqual([self.state1], history)

        machine.evolve()
        machine.commit()

        score, history = machine.optimal_path()

        self.assertEqual(- np.log(t1_prob) - np.log(t2_prob), score)
        self.assertEqual([self.state1, self.state1], history)

    def test_multiple_states_model(self):
        machine = Graph(self.states)
        self.states[0].add_transition(
            Transition.free(self.states[0], self.states[1])
        )
        self.states[1].add_transition(
            Transition.free(self.states[1], self.states[1])
        )

        machine.evolve()
        machine.commit()

        score, history = machine.optimal_path()

        self.assertEqual([self.state1], history)
        self.assertEqual(- np.log(self.p1[0]), score)

        machine.evolve()
        machine.commit()

        score, history = machine.optimal_path()

        self.assertEqual([self.state1, self.state2], history)
        self.assertEqual(- np.log(self.p1[0]) - np.log(self.p2[1]), score)

    def test_word_transition(self):
        graph_a = Graph([self.states[0]])
        graph_b = Graph([self.states[1]])

        a_to_b = Transition(graph_a, graph_b, 0.25)

        graph_a.add_transition(a_to_b)

        top_level = Graph([graph_a, graph_b])
        top_level.evolve()
        top_level.commit()
        score, history = top_level.optimal_path()

        self.assertEqual(graph_a.optimal_path()[0] + a_to_b.full_cost(), score)
        self.assertEqual([], history)


# todo: composite pattern: SentenceModel.passToken(), WordModel.passToken(src, dest), Node.passToken(src, dest)
"""
def iteration(self)
    for state in states:
        out_states = model.outgoing(state)
        for out_state in out_states:
            model.passToken(state, out_state)
"""
