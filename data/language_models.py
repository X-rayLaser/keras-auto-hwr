from nltk.tokenize import word_tokenize


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

        return WordDictionary(words, transitions_tuples, None)

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
    def build(provider):
        return TransitionsBuilder(provider).build()

    def __init__(self, words, transitions, text_encoder):
        self.words = words
        self.encoder = text_encoder
        self.transitions = transitions

    def encoded(self, index):
        text = self.words[index]
        return [self.encoder.encode(ch) for ch in text]

    def __len__(self):
        return len(self.words)
