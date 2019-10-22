import numpy as np


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
        self._flush_staging()

    def _flush_staging(self):
        self._staging = (self.infinite_score, [])

    def initialize_token(self):
        self._score = 0

    def pass_token(self, token):
        score, history = token
        history = history + [self.state]
        token = (score, history)

        current_score, history = self._staging
        if score < current_score:
            self._staging = token

    def commit(self):
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
        new_history = history
        self.destination.pass_token((new_score, new_history))


class Graph(Node):
    def __init__(self, nodes):
        initial_state = NullState()
        self._nodes = nodes + [initial_state]

        for node in nodes:
            transition = Transition.free(initial_state, node)
            initial_state.add_transition(transition)

        self._transitions = []

    def local_cost(self):
        return self._nodes[0].local_cost()

    @property
    def token(self):
        last_node = self._nodes[-2]
        return last_node.token

    @property
    def transitions(self):
        return self._transitions

    def commit(self):
        for node in self._nodes:
            node.commit()

    def pass_token(self, token):
        self._nodes[0].pass_token(token)

    def add_transition(self, transition):
        self._transitions.append(transition)

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