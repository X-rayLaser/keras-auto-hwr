import numpy as np


class Node:
    def __init__(self):
        self._transitions = []

    def evolve(self):
        pass

    def optimal_path(self):
        raise NotImplementedError

    def commit(self):
        raise NotImplementedError

    def pass_token(self, token):
        raise NotImplementedError

    def add_transition(self, transition):
        self._transitions.append(transition)

    @property
    def transitions(self):
        return self._transitions


class State(Node):
    infinite_score = np.inf

    def __init__(self, state, probabilities):
        super().__init__()
        self.state = state
        self.p = probabilities

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
    def token(self):
        return self._score, self._history

    def optimal_path(self):
        return self.token


class Transition:
    @staticmethod
    def free(source, destination):
        return Transition(source, destination, 1.0)

    def __init__(self, source, destination, probability):
        self._source = source
        self.destination = destination
        self._p = probability

    def cost(self):
        return - np.log(self._p)

    def full_cost(self):
        return self.cost() + self.destination.local_cost()

    def pass_token(self):
        score = self.full_cost()
        old_score, history = self._source.token
        print(old_score, history)
        new_score = old_score + score
        new_history = history

        print('new token',new_score, new_history)
        self.destination.pass_token((new_score, new_history))


class Graph(Node):
    def __init__(self, nodes):
        super().__init__()
        initial_state = NullState()
        self._nodes = nodes + [initial_state]

        for node in nodes:
            transition = Transition.free(initial_state, node)
            self.add_transition(transition)

    def local_cost(self):
        return self._nodes[0].local_cost()

    @property
    def token(self):
        last_node = self._nodes[-2]
        return last_node.token

    def commit(self):
        for node in self._nodes:
            node.commit()

    def pass_token(self, token):
        self._nodes[0].pass_token(token)

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

        for transition in self.transitions:
            transition.pass_token()


class NullState(Node):
    def __init__(self):
        super().__init__()
        self._step = 0

    def commit(self):
        self._step += 1

    def pass_token(self, token):
        pass

    def local_cost(self):
        return State.infinite_score

    @property
    def token(self):
        print('step', self._step)
        if self._step == 0:
            return 0, []

        return State.infinite_score, []

    def optimal_path(self):
        return self.token
