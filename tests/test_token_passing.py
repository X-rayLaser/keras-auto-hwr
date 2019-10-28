from unittest import TestCase
import numpy as np
from algorithms.token_passing import State, Transition, Graph, Token, WordModelFactory, TokenPassing
from data.language_models import WordDictionary
from data.encodings import CharacterTable


class TokenBaseTests(TestCase):
    def setUp(self):
        self.score = 25
        self.history = [15, 30]
        self.words = [3, 4]


class TokenTests(TokenBaseTests):
    def test_equality(self):
        a = Token(self.score, self.history, self.words)
        b = Token(self.score, self.history, self.words)

        self.assertEqual(a, b)
        self.assertEqual(a, a)

    def test_inequality(self):
        self.assertNotEqual(Token(self.score, self.history, self.words),
                            Token(self.score, self.history, [3]))

        self.assertNotEqual(Token(self.score, self.history, self.words),
                            Token(self.score, [96], self.words))

        self.assertNotEqual(Token(self.score, self.history, self.words),
                            Token(self.score + 5, self.history, self.words))

    def test_updated_returns_correct_token(self):
        token = Token(self.score, self.history)

        res = token.updated(5, 80)
        expected = Token(30, self.history + [80])
        self.assertEqual(expected, res)

    def test_update_score(self):
        token = Token(self.score, self.history)
        res = token.update_score(5)
        self.assertEqual(Token(self.score + 5, self.history), res)
        self.assertEqual(Token(self.score, self.history), token)

    def test_update_history(self):
        token = Token(self.score, self.history, self.words)
        res = token.update_history(42)
        self.assertEqual(Token(self.score, self.history + [42], self.words), res)

    def test_update_words(self):
        token = Token(self.score, self.history, self.words)
        res = token.update_words(42)
        self.assertEqual(Token(self.score, self.history, self.words + [42]), res)


class TokenDeepCopyTests(TokenBaseTests):
    def setUp(self):
        super().setUp()

        self.history_copy = list(self.history)
        self.words_copy = list(self.words)
        self.token = Token(self.score, self.history_copy, self.words_copy )

    def modify_token_content(self, token):
        token.history.append(40)
        token.words.append(3)

    def did_not_changed(self):
        self.assertEqual(self.history, self.history_copy)
        self.assertEqual(self.words, self.words_copy)

    def test_update_score_create_deep_copy(self):
        token = self.token.update_score(10)
        self.modify_token_content(token)
        self.did_not_changed()

    def test_update_history_create_deep_copy(self):
        token = self.token.update_history(10)
        self.modify_token_content(token)
        self.did_not_changed()

    def test_update_words_create_deep_copy(self):
        token = self.token.update_words(10)
        self.modify_token_content(token)
        self.did_not_changed()


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

        self.assertEqual(Token(expected_score, [self.state.state]),
                         self.state.token)


class StateTests(TestCase):
    def setUp(self):
        self.state_value = 25
        self.p = [0.5, 0.2, 0.1]

    def test_pass_token_retains_words(self):
        state = State(self.state_value, self.p)
        words = [5, 10]
        token = Token(25, [], words)
        state.pass_token(token)
        state.commit()
        self.assertEqual(words, state.token.words)

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
        self.assertEqual(Token(state.infinite_score, []), state.token)

    def test_token_after_calling_initialization_method(self):
        state = State(self.state_value, self.p)
        state.initialize_token()
        self.assertEqual(Token(0, []), state.token)

    def test_pass_token_with_transit_cost(self):
        state = State(self.state_value, self.p)
        local_cost = state.local_cost()

        token = Token(45.3, [state])
        state.pass_token(token, transit_cost=10)
        state.pass_token(Token(20.3, []), transit_cost=2)

        state.commit()
        self.assertEqual(Token(20.3 + local_cost + 2, [self.state_value]), state.token)

    def test_token_after_passing_token_and_commit(self):
        state = State(self.state_value, self.p)
        local_cost = state.local_cost()

        token = Token(45.3, [state])
        state.pass_token(token)
        state.pass_token(Token(20.3, []))

        state.commit()

        self.assertEqual(Token(20.3 + local_cost, [self.state_value]),
                         state.token)

    def test_before_commit(self):
        state = State(self.state_value, self.p)

        token = Token(45.3, [])
        state.pass_token(token)

        self.assertEqual(Token(state.infinite_score, []), state.token)

    def test_pass_token_after_commit(self):
        state = State(self.state_value, self.p)
        token = Token(45.3, [state])
        state.pass_token(token)
        state.commit()

        cost = state.local_cost()
        state.pass_token(Token(80.3, []))
        state.pass_token(Token(50.3, []))
        state.pass_token(Token(90.3, []))

        state.commit()

        self.assertEqual(Token(50.3 + cost, [self.state_value]), state.token)

    def test_commit_without_passing_token(self):
        state = State(self.state_value, self.p)
        token = Token(45.3, [30])
        state.pass_token(token)
        state.commit()

        state.commit()

        self.assertEqual(Token(state.infinite_score, []), state.token)


class GraphTests(TestCase):
    def setUp(self):
        self.state1 = 50
        self.state2 = 20

        self.p1 = [0.5, 0.1]
        self.p2 = [0.1, 0.2]
        self.states = [State(self.state1, self.p1), State(self.state2, self.p2)]

    def test_passing_token_to_graph_adds_word_id_to_history(self):
        graph = Graph(self.states, graph_id=7)
        token = Token(25, [3], words=[3])
        graph.pass_token(token)
        graph.commit()
        self.assertEqual([3, 7], self.states[0].token.words)

    def test_local_cost(self):
        graph = Graph(self.states)
        self.assertEqual(- np.log(self.p1[0]), graph.local_cost())

    def test_token(self):
        graph = Graph(self.states)
        self.states[1].pass_token(Token(34, [20, 30]))
        self.states[1].commit()
        self.assertEqual(self.states[1].token, graph.token)

    def test_token_on_nested_graph(self):
        g1 = Graph([self.states[0]])
        g2 = Graph([self.states[1]])

        g3_state = State(34, [0.2])
        g3 = Graph([g3_state])
        top_level = Graph([g1, g2, g3])

        token = Token(3, [23])
        expected_score = 3 + g3.local_cost()
        g3_state.pass_token(token)
        g3_state.commit()

        self.assertEqual(expected_score, top_level.token.score)

    def test_pass_token(self):
        graph = Graph(self.states)
        score = 30
        expected_score = score + graph.local_cost()
        expected_history = [20, 30, self.state1]
        graph.pass_token(Token(score, expected_history[:-1]))
        graph.commit()
        first_state = self.states[0]

        self.assertEqual(expected_score, first_state.token.score)
        self.assertEqual(expected_history, first_state.token.history)

    def test_best_path_initially(self):
        graph = Graph(self.states)
        self.states[0].add_transition(
            Transition.free(self.states[0], self.states[1])
        )
        self.states[1].add_transition(
            Transition.free(self.states[1], self.states[1])
        )

        self.assertEqual([], graph.optimal_path().history)

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

        self.assertEqual(Token(- np.log(t1_prob), [self.state1]),
                         graph.optimal_path())

        graph.evolve()
        graph.commit()

        expected = Token(- np.log(t1_prob) - np.log(t2_prob),
                         [self.state1, self.state1])
        self.assertEqual(expected, graph.optimal_path())

    def test_multiple_states_model(self):
        graph_id = 7
        graph = Graph(self.states, graph_id=graph_id)
        graph.add_transition(
            Transition.free(self.states[0], self.states[1])
        )
        graph.add_transition(
            Transition.free(self.states[1], self.states[1])
        )

        graph.evolve()
        graph.commit()

        self.assertEqual(Token(- np.log(self.p1[0]), [self.state1], [graph_id]),
                         graph.optimal_path())

        graph.evolve()
        graph.commit()

        expected = Token(- np.log(self.p1[0]) - np.log(self.p2[1]),
                         [self.state1, self.state2], [graph_id])
        self.assertEqual(expected, graph.optimal_path())

    def test_word_transition(self):
        graph_a = Graph([self.states[0]])
        graph_b = Graph([self.states[1]])

        a_to_b = Transition(graph_a, graph_b, 0.99)

        graph_a.add_transition(a_to_b)

        top_level = Graph([graph_a, graph_b])
        top_level.evolve()
        top_level.commit()

        self.assertEqual(graph_a.token, top_level.optimal_path())

        transit_cost = a_to_b.full_cost()

        top_level.evolve()
        top_level.commit()
        token = top_level.optimal_path()

        self.assertEqual(- np.log(0.5) + transit_cost, token.score)
        self.assertEqual([self.state1, self.state2], token.history)

    def test_transitions_with_self_loops(self):
        first_state = State(20, [0.2, 0.4])
        graph = Graph([first_state])

        graph.add_transition(Transition.free(first_state, first_state))

        graph.evolve()
        graph.commit()

        graph.evolve()
        graph.commit()
        score = graph._nodes[0].token.score

        self.assertEqual(- np.log(0.2) - np.log(0.4), score)


class WordModelFactoryTests(TestCase):
    def setUp(self):
        self.step1_pmf = [0.3, 0.7]
        self.step2_pmf = [0.1, 0.9]
        self.distribution = np.array([self.step1_pmf, self.step2_pmf])
        self.factory = WordModelFactory(self.distribution)

    def test_transitions_are_in_place(self):
        model = self.factory.create_model([1, 0])

        self.assertEqual(5, len(model.transitions))

        self.assertEqual(model.transitions[2]._source.state, 1)
        self.assertEqual(model.transitions[2].destination.state, 1)

        self.assertEqual(model.transitions[3]._source.state, 0)
        self.assertEqual(model.transitions[3].destination.state, 0)

        self.assertEqual(model.transitions[4]._source.state, 1)
        self.assertEqual(model.transitions[4].destination.state, 0)

    def test_correct_emission_probabilities_applied(self):
        model = self.factory.create_model([0, 1])

        model.evolve()
        model.commit()
        self.assertEqual(- np.log(self.step1_pmf[0]), model._nodes[0].token.score)
        self.assertEqual(- np.log(self.step1_pmf[1]), model._nodes[1].token.score)

        model.evolve()
        model.commit()

        self.assertEqual(- np.log(self.step1_pmf[0]) - np.log(self.step2_pmf[0]),
                         model._nodes[0].token.score)
        self.assertEqual(- np.log(self.step1_pmf[1]) - np.log(self.step2_pmf[1]),
                         model._nodes[1].token.score)

    def test_model_contains_correct_words_history(self):
        model = self.factory.create_model([0, 1], model_id=45)
        self.assertEqual(45, model.graph_id)


class TokenPassingTests(TestCase):
    def setUp(self):
        transitions = {
            ("hello", "world"): 0.75,
            ("hello", "hello"): 0.25,
            ("world", "hello"): 0.3,
            ("world", "world"): 0.7,
        }

        self.char_table = CharacterTable()
        self.hello_code = 0
        self.world_code = 1
        self.dictionary = WordDictionary(["hello", "world"], transitions)

    def create_distribution(self, text):
        char_table = CharacterTable()

        codes = [char_table.encode(ch) for ch in text]

        Tx = len(text)
        n = len(char_table)
        a = np.zeros((Tx, n))
        for i, code in enumerate(codes):
            a[i, code] = 1.0

        return a

    def test_with_hello(self):
        distribution = self.create_distribution('hhhheeellllllo')

        decoder = TokenPassing(self.dictionary, distribution, self.char_table)
        res = decoder.decode()
        self.assertEqual([self.hello_code], res)

    def test_with_world(self):
        distribution = self.create_distribution('wworrrlldddd')

        decoder = TokenPassing(self.dictionary, distribution, self.char_table)
        res = decoder.decode()
        self.assertEqual([self.world_code], res)

    def test_with_hello_world(self):
        distribution = self.create_distribution('hheellloowworrlldd')

        decoder = TokenPassing(self.dictionary, distribution, self.char_table)
        res = decoder.decode()
        self.assertEqual([self.hello_code, self.world_code], res)

    def test_with_world_hello(self):
        distribution = self.create_distribution('wwwoorrlddhello')

        decoder = TokenPassing(self.dictionary, distribution, self.char_table)
        res = decoder.decode()
        self.assertEqual([self.world_code, self.hello_code], res)

    def test_with_repeating_word(self):
        distribution = self.create_distribution('hellohello')

        decoder = TokenPassing(self.dictionary, distribution, self.char_table)
        res = decoder.decode()
        self.assertEqual([self.hello_code, self.hello_code], res)

    def test_partial_match(self):
        distribution = self.create_distribution('hheel')

        decoder = TokenPassing(self.dictionary, distribution, self.char_table)
        res = decoder.decode()
        self.assertEqual([self.hello_code], res)


# todo: word model should contain blanks
# todo: each word may allow to start and with punctuation sign, space or blank
# todo: few more edge cases
