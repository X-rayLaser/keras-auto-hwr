from nltk.tokenize import word_tokenize
import json


class BaseTokenizer:
    def word_tokenize(self, text, language='english'):
        return word_tokenize(text, language=language)


class TransitionsBuilder:
    def __init__(self, provider, tokenizer=None):
        self._provider = provider

        if tokenizer:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = BaseTokenizer()

    def build(self):
        words = self.extract_words()

        gen = self.sentences_generator()
        transitions_dict = self.build_transitions(words, gen)

        transitions_tuples = {}
        for first_word in transitions_dict.keys():
            pmf = transitions_dict[first_word]
            for second_word, prob in pmf.items():
                transitions_tuples[(first_word, second_word)] = prob

        return WordDictionary(words, transitions_tuples)

    def build_transitions(self, words, sentences_generator):
        res = self.initialize_dictionary(words)

        for sentence in sentences_generator:
            for previous, current in zip(sentence, sentence[1:]):
                pmf = res[previous]
                pmf[current] += 1

        self.normalize_probabilities(res)

        return res

    def normalize_probabilities(self, prob_dict):
        for word in prob_dict.keys():
            pmf = prob_dict[word]
            total = sum([v for v in pmf.values()])
            if total > 0:
                for k, v in pmf.items():
                    pmf[k] = pmf[k] / total

    def initialize_dictionary(self, words):
        res = {}
        for i in range(len(words)):
            res[words[i]] = {}
            for j in range(len(words)):
                res[words[i]][words[j]] = 0
        return res

    def extract_words(self):
        res = set()
        for words in self.sentences_generator():
            res = res.union(words)
        words = list(res)
        return words

    def sentences_generator(self):
        for _, ys in self._provider.get_sequences():
            yield self._tokenizer.word_tokenize(ys)


class WordDictionary:
    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            s = f.read()

        d = json.loads(s)

        words = d['words']

        nested_dict = d['transitions']
        transitions = WordDictionary.to_tuples_dict(nested_dict)

        return WordDictionary(words, transitions)

    @staticmethod
    def build(provider):
        return TransitionsBuilder(provider).build()

    def __init__(self, words, transitions):
        self.words = words
        self.transitions = transitions

        self.word_to_index = {}
        for i, word in enumerate(words):
            self.word_to_index[word] = i

    def transition_p(self, src, dest):
        if (src, dest) not in self.transitions:
            return 0

        return self.transitions[(src, dest)]

    def encoded(self, index, text_encoder):
        text = self.words[index]
        return [text_encoder.encode(ch) for ch in text]

    def __len__(self):
        return len(self.words)

    def save(self, path):
        d = {
            'words': self.words,
            'transitions': self.to_nested_dict(self.transitions)
        }

        s = json.dumps(d)

        with open(path, 'w') as f:
            f.write(s)

    def word_index(self, word):
        return self.word_to_index[word]

    @staticmethod
    def to_nested_dict(transitions):
        d = {}
        for tup, p in transitions.items():
            a, b = tup
            if a not in d:
                d[a] = {}

            d[a][b] = p

        return d

    @staticmethod
    def to_tuples_dict(transitions):
        res = {}
        for first_word in transitions.keys():
            d = transitions[first_word]
            for second_word, p in d.items():
                res[(first_word, second_word)] = p

        return res
