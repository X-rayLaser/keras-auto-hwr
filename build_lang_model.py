import nltk
from nltk.corpus import webtext
from data.language_models import WordDictionary


class DictionaryBuilder:
    def build(self, max_words):
        bigram_freqs = self.extract_word_bigrams(max_words)

        all_words = self.extract_words(bigram_freqs)

        d = self.to_nested_dict(bigram_freqs)

        transitions = self.to_probabilities(d)

        return WordDictionary(list(all_words), transitions)

    def get_file_ids(self):
        try:
            file_ids = webtext.fileids()
        except LookupError:
            nltk.download('webtext')
            file_ids = webtext.fileids()
        return file_ids

    def extract_word_bigrams(self, max_words):
        file_ids = self.get_file_ids()

        freq_dict = nltk.FreqDist([])

        for file_id in file_ids:
            words = [word for word in webtext.words(file_id) if word[0].isalnum()]
            freq_dict = freq_dict + nltk.FreqDist(nltk.bigrams(words))

        return freq_dict.most_common(max_words)

    def extract_words(self, bi_grams):
        all_words = set()

        for bi_gram, freq in bi_grams:
            word_from, word_to = bi_gram

            all_words.add(word_from)
            all_words.add(word_to)

        return list(all_words)

    def to_nested_dict(self, bigram_freqs):
        d = {}

        for bi_gram, freq in bigram_freqs:
            word_a, word_b = bi_gram
            if word_a not in d:
                d[word_a] = {}

            d[word_a][word_b] = freq

        return d

    def to_probabilities(self, nested_dict):
        transitions = {}
        for first_word in list(nested_dict.keys()):
            counts = nested_dict[first_word]

            total = sum(counts.values())

            for second_word, count in dict(counts).items():
                transitions[(first_word, second_word)] = count / total

        return transitions


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    #parser.add_argument('corpora', type=str)
    parser.add_argument('destination', type=str)
    parser.add_argument('--lang', type=str, default='eng')
    parser.add_argument('--max_words', type=int, default=10000)

    args = parser.parse_args()

    builder = DictionaryBuilder()
    word_dictionary = builder.build(args.max_words)
    word_dictionary.save(args.destination)


# todo adopt word dictionary builder class so as to allow using lines generator
# todo furthermore, add text encoding parameter
# todo perform this step automatically during compilation step
# todo factories for predictors
# todo pass lang model path to the ocrdemo script
# todo generalize to other languages
# todo test code for building language model
