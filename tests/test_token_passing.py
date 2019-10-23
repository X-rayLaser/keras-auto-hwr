from unittest import TestCase
import numpy as np
from algorithms.token_passing import State, Transition, Graph


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
        self.assertEqual([self.state_value], history)

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
        self.assertEqual([self.state_value], history)

    def test_commit_without_passing_token(self):
        state = State(self.state_value, self.p)
        token = (45.3, [30])
        state.pass_token(token)
        state.commit()

        state.commit()

        score, history = state.token
        self.assertEqual(state.infinite_score, score)
        self.assertEqual([], history)


class GraphTests(TestCase):
    def setUp(self):
        self.state1 = 50
        self.state2 = 20

        self.p1 = [0.5, 0.1]
        self.p2 = [0.1, 0.2]
        self.states = [State(self.state1, self.p1), State(self.state2, self.p2)]

    def test_local_cost(self):
        graph = Graph(self.states)
        self.assertEqual(- np.log(self.p1[0]), graph.local_cost())

    def test_token(self):
        graph = Graph(self.states)
        self.states[1].pass_token((34, [20, 30]))
        self.states[1].commit()
        self.assertEqual(self.states[1].token, graph.token)

    def test_token_on_nested_graph(self):
        g1 = Graph([self.states[0]])
        g2 = Graph([self.states[1]])

        g3_state = State(34, [0.2])
        g3 = Graph([g3_state])
        top_level = Graph([g1, g2, g3])

        token = (3, [23])
        g3_state.pass_token(token)
        g3_state.commit()

        score, history = top_level.token
        self.assertEqual(3, score)

    def test_pass_token(self):
        graph = Graph(self.states)
        expected_score = 34
        expected_history = [20, 30, self.state1]
        graph.pass_token((expected_score, expected_history[:-1]))
        graph.commit()
        first_state = self.states[0]
        score, history = first_state.token
        self.assertEqual(expected_score, score)
        self.assertEqual(expected_history, history)

    def test_best_path_initially(self):
        graph = Graph(self.states)
        self.states[0].add_transition(
            Transition.free(self.states[0], self.states[1])
        )
        self.states[1].add_transition(
            Transition.free(self.states[1], self.states[1])
        )

        score, history = graph.optimal_path()
        self.assertEqual([], history)

    def test_one_state_model(self):
        states = [self.states[0]]
        graph = Graph(states)
        graph.add_transition(
            Transition.free(self.states[0], self.states[0])
        )

        graph.evolve()
        graph.commit()

        t1_prob = self.p1[0]
        t2_prob = self.p1[1]
        score, history = graph.optimal_path()

        self.assertEqual(- np.log(t1_prob), score)
        self.assertEqual([self.state1], history)

        graph.evolve()
        graph.commit()

        score, history = graph.optimal_path()

        self.assertEqual(- np.log(t1_prob) - np.log(t2_prob), score)
        self.assertEqual([self.state1, self.state1], history)

    def test_multiple_states_model(self):
        graph = Graph(self.states)
        graph.add_transition(
            Transition.free(self.states[0], self.states[1])
        )
        graph.add_transition(
            Transition.free(self.states[1], self.states[1])
        )

        graph.evolve()
        graph.commit()

        score, history = graph.optimal_path()

        self.assertEqual([self.state1], history)
        self.assertEqual(- np.log(self.p1[0]), score)

        graph.evolve()
        graph.commit()

        score, history = graph.optimal_path()

        self.assertEqual([self.state1, self.state2], history)
        self.assertEqual(- np.log(self.p1[0]) - np.log(self.p2[1]), score)

    def test_word_transition(self):
        graph_a = Graph([self.states[0]])
        graph_b = Graph([self.states[1]])

        a_to_b = Transition(graph_a, graph_b, 0.99)

        graph_a.add_transition(a_to_b)

        top_level = Graph([graph_a, graph_b])
        top_level.evolve()
        top_level.commit()
        score, history = top_level.optimal_path()

        self.assertEqual(graph_a.token[0], score)
        self.assertEqual(graph_a.token[1], history)

        transit_cost = a_to_b.full_cost()

        top_level.evolve()
        top_level.commit()
        score, history = top_level.optimal_path()
        self.assertEqual([self.state1, self.state2], history)
        self.assertEqual(- np.log(0.5) + transit_cost, score)

    def test_transitions_with_self_loops(self):
        first_state = State(20, [0.2, 0.4])
        graph = Graph([first_state])

        graph.add_transition(Transition.free(first_state, first_state))

        print('evolve')
        graph.evolve()
        graph.commit()

        print('evolve')
        graph.evolve()
        graph.commit()
        score, _ = graph._nodes[0].token

        self.assertEqual(- np.log(0.2) - np.log(0.4), score)


class WordModelFactory:
    def __init__(self, distribution):
        self._pmfs = distribution

    def create_model(self, char_codes):
        states = []
        for code in char_codes:
            p = self._pmfs[:, code]
            states.append(State(code, p))

        graph = Graph(states)

        for state in states:
            graph.add_transition(Transition.free(state, state))

        for current, previous in zip(states[1:], states):
            graph.add_transition(Transition.free(previous, current))

        return graph


class WordModelFactoryTests(TestCase):
    def test_transitions_are_in_place(self):
        distribution = np.array([
            [0.3, 0.7], [0.1, 0.9]
        ])

        factory = WordModelFactory(distribution)

        model = factory.create_model([1, 0])

        self.assertEqual(5, len(model.transitions))

        self.assertEqual(model.transitions[2]._source.state, 1)
        self.assertEqual(model.transitions[2].destination.state, 1)

        self.assertEqual(model.transitions[3]._source.state, 0)
        self.assertEqual(model.transitions[3].destination.state, 0)

        self.assertEqual(model.transitions[4]._source.state, 1)
        self.assertEqual(model.transitions[4].destination.state, 0)

    def test_correct_emission_probabilities_applied(self):
        step1_pmf = [0.3, 0.7]
        step2_pmf = [0.1, 0.9]

        distribution = np.array([step1_pmf, step2_pmf])

        factory = WordModelFactory(distribution)
        model = factory.create_model([0, 1])

        model.evolve()
        model.commit()
        score, _ = model._nodes[0].token
        self.assertEqual(- np.log(0.3), score)
        score, _ = model._nodes[1].token
        self.assertEqual(- np.log(0.7), score)

        model.evolve()
        model.commit()
        score, _ = model._nodes[0].token

        self.assertEqual(- np.log(0.3) - np.log(0.1), score)
        score, _ = model._nodes[1].token
        self.assertEqual(- np.log(0.7) - np.log(0.9), score)


class WordDictionary:
    def __init__(self, words, text_encoder):
        self.words = words
        self.encoder = text_encoder

    def encoded(self, index):
        text = self.words[index]
        return [self.encoder.encode(ch) for ch in text]

    def __len__(self):
        return len(self.words)


class TokenPassing:
    def __init__(self, dictionary, distribution):
        graphs = []
        for i in range(len(dictionary)):
            codes = dictionary.encoded(i)

            graphs.append(Graph())

    def decode(self):
        return []


class TokenPassingTests(TestCase):
    def create_distribution(self, text):
        from data.encodings import CharacterTable
        char_table = CharacterTable()

        codes = [char_table.encode(ch) for ch in text]

        import numpy as np

        Tx = len(text)
        n = len(char_table)
        a = np.zeros((Tx, n))
        for i, code in enumerate(codes):
            a[i, code] = 1.0

        return a

    def test(self):

        transitions = {
            ("hello", "world"): 0.75,
            ("hello", "hello"): 0.25,
            ("world", "hello"): 0.3,
            ("world", "world"): 0.7,
        }
        dictionary = WordDictionary(["hello", "world"], transitions)

        distribution = self.create_distribution('hheellloowworrlldd')

        decoder = TokenPassing(dictionary, distribution)
        res = decoder.decode()
        self.assertEqual([0, 1], res)


# todo: make graph return path through words instead of characters
# todo: word model, algorithm
# todo: few more edge cases
