from unittest import TestCase
from data.language_models import WordDictionary
from data.providers import BaseSource


class ProviderMock(BaseSource):
    def __init__(self, sentences):
        self.sentences = sentences

    def get_sequences(self):
        for s in self.sentences:
            yield [], s

    def __len__(self):
        return len(self.sentences)


class WordDictionaryTests(TestCase):
    def test_create_model_from_one_word(self):
        provider = ProviderMock(['First'])

        dictionary = WordDictionary.build(provider)

        self.assertEqual(['First'], dictionary.words)

        self.assertEqual({('First', 'First'): 0}, dictionary.transitions)

    def test_create_model_from_short_sentence(self):
        provider = ProviderMock(['Short sentence'])

        dictionary = WordDictionary.build(provider)

        self.assertEqual({'Short', 'sentence'}, set(dictionary.words))

        expected_transitions = {
            ('Short', 'Short'): 0.0,
            ('sentence', 'Short'): 0.0,
            ('Short', 'sentence'): 1.0,
            ('sentence', 'sentence'): 0.0,
        }
        self.maxDiff = None
        self.assertEqual(expected_transitions, dictionary.transitions)

    def test_create_model_from_few_sentences(self):
        first_word = 'First'
        second_word = 'second'
        provider = ProviderMock([first_word, second_word])

        dictionary = WordDictionary.build(provider)

        self.assertEqual({first_word, second_word}, set(dictionary.words))

        expected_transitions = {
            (first_word, first_word): 0,
            (first_word, second_word): 0,
            (second_word, first_word): 0,
            (second_word, second_word): 0
        }
        self.assertEqual(expected_transitions, dictionary.transitions)

    def test_create_model_from_repeating_words(self):
        hello = 'hello'
        world = 'world'
        there = 'there'

        provider = ProviderMock(['{} {} {} {}'.format(hello, world, hello, there)])

        dictionary = WordDictionary.build(provider)
        self.assertEqual({hello, world, there}, set(dictionary.words))

        expected_transitions = {
            (hello, hello): 0,
            (hello, world): 0.5,
            (hello, there): 0.5,
            (world, hello): 1.0,
            (world, world): 0,
            (world, there): 0,
            (there, hello): 0,
            (there, world): 0,
            (there, there): 0,
        }

        self.assertEqual(expected_transitions, dictionary.transitions)


# todo: test with non english words
# todo: save/load language model
# todo: build from text corpora
