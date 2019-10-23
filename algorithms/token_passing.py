import numpy as np


class Token:
    def __init__(self, score, history, words=None):
        self.score = score
        self.history = history

        if words is None:
            self.words = []
        else:
            self.words = words

    def __eq__(self, other):
        return (self.score == other.score and self.history == other.history
                and self.words == other.words)

    def updated(self, cost=None, node_id=None, new_word=None):
        if cost is None:
            score = self.score
        else:
            score = self.score + cost

        if node_id is None:
            history = list(self.history)
        else:
            history = self.history + [node_id]

        if new_word is None:
            words = list(self.words)
        else:
            words = self.words + [new_word]

        return Token(score, history, words)

    def update_score(self, cost):
        return Token(self.score + cost, list(self.history), list(self.words))

    def update_history(self, node_id):
        return Token(self.score, self.history + [node_id], list(self.words))

    def update_words(self, new_word_id):
        return Token(self.score, list(self.history), self.words + [new_word_id])

    def __repr__(self):
        return '(score: {}, history: {}, words: {})'.format(
            self.score, self.history, self.words
        )


class Node:
    def __init__(self):
        self._transitions = []

    def evolve(self):
        pass

    def optimal_path(self):
        raise NotImplementedError

    def commit(self):
        raise NotImplementedError

    def pass_token(self, token, transit_cost=0):
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

        self._token = Token(self.infinite_score, [])
        self._score = self.infinite_score
        self._history = []
        self._words = []

        self._staging = None
        self._step = 0
        self._flush_staging()

    def _flush_staging(self):
        self._staging = Token(self.infinite_score, [])

    def initialize_token(self):
        self._score = 0

    def pass_token(self, token, transit_cost=0):
        total_cost = transit_cost + self.local_cost()
        new_token = token.update_score(total_cost).update_history(self.state)

        if new_token.score < self._staging.score:
            self._staging = new_token

    def commit(self):
        self._score = self._staging.score
        self._history = self._staging.history
        self._words = self._staging.words
        self._flush_staging()
        self._step += 1

    def local_cost(self):
        index = self._step
        return - np.log(self.p[index])

    @property
    def token(self):
        return Token(self._score, self._history, self._words)

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
        self.destination.pass_token(self._source.token, self.cost())


class Graph(Node):
    def __init__(self, nodes, graph_id=0):
        super().__init__()
        self._graph_id = graph_id
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

    def pass_token(self, token, transit_cost=0):
        token = token.update_words(self._graph_id)
        self._nodes[0].pass_token(token, transit_cost)

    def optimal_path(self):
        best_token = Token(np.inf, self._nodes[0].token.history)
        for state in self._nodes:
            token = state.optimal_path()
            if token.score < best_token.score:
                best_token = token

        return best_token

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

    def pass_token(self, token, transit_cost=0):
        pass

    def local_cost(self):
        return State.infinite_score

    @property
    def token(self):
        if self._step == 0:
            return Token(0, [])

        return Token(State.infinite_score, [])

    def optimal_path(self):
        return self.token
