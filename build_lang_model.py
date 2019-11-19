import nltk
from nltk.corpus import webtext
from data.language_models import WordDictionary
import os
from data.data_set_home import DataSetHome, create_deterministic_source


class DictionaryBuilder:
    def build(self, max_words):
        bigram_freqs, most_common_words = self.extract_word_bigrams(max_words)

        d = self.to_nested_dict(bigram_freqs)

        transitions = self.to_probabilities(d)

        return WordDictionary(list(most_common_words), transitions)

    def get_file_ids(self):
        try:
            file_ids = webtext.fileids()
        except LookupError:
            nltk.download('webtext')
            file_ids = webtext.fileids()
        return file_ids

    def extract_word_bigrams(self, max_words):
        file_ids = self.get_file_ids()

        all_words = []
        for file_id in file_ids:
            words = [word for word in webtext.words(file_id) if word[0].isalnum()]
            all_words.extend(words)

        most_common_words = set([w for w, _ in nltk.FreqDist(all_words).most_common(max_words)])

        bigrams = []
        for w_from, w_to in nltk.bigrams(all_words):
            if w_from in most_common_words and w_to in most_common_words:
                bigrams.append((w_from, w_to))

        return nltk.FreqDist(bigrams).most_common(), most_common_words

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


def save_dictionary(word_dictionary, dict_path):
    with open(dict_path, 'w') as f:

        for i in range(len(word_dictionary)):
            codes = word_dictionary.encoded(i, encoding_table)

            blank = len(encoding_table)
            with_blanks = [blank]
            for code in codes:
                with_blanks.append(code)
                with_blanks.append(blank)

            s = ' '.join(map(str, with_blanks))
            f.write(s + '\n')


def save_bigrams(word_dictionary, bigrams_path):
    d = word_dictionary.to_nested_dict(word_dictionary.transitions)

    with open(bigrams_path, 'w') as f:
        num_sources = len(d.keys())
        f.write('{}\n'.format(num_sources))

        for from_word in d.keys():
            from_index = word_dictionary.word_index(from_word)
            num_destinations = len(d[from_word].keys())

            f.write('{} {}\n'.format(from_index, num_destinations))

            for to_word, p in d[from_word].items():
                to_index = word_dictionary.word_index(to_word)
                f.write('{} {}\n'.format(to_index, p))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=str)
    parser.add_argument('--lang', type=str, default='eng')
    parser.add_argument('--max_words', type=int, default=10000)

    args = parser.parse_args()

    builder = DictionaryBuilder()
    word_dictionary = builder.build(args.max_words)
    word_dictionary.save('english.json')

    ds_home = os.path.join(os.getcwd(), 'compiled', 'ds1')
    home = DataSetHome(ds_home, create_deterministic_source)
    encoding_table = home.get_encoding_table()

    dict_location = args.destination

    dict_path = os.path.join(dict_location, 'dictionary.txt')
    bigrams_path = os.path.join(dict_location, 'bigrams.txt')

    save_dictionary(word_dictionary, dict_path)
    save_bigrams(word_dictionary, bigrams_path)


# todo perform this step automatically during compilation step
# todo factories for predictors
# todo pass lang model path to the ocrdemo script
# todo generalize to other languages
# todo test code for building language model
