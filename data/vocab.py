from data.char_table import CharacterTable
from collections import Counter


class Vocabulary(CharacterTable):
    def __init__(self, word_set, max_size=1000):
        super().__init__()
        self._word_set = word_set
        self._max_size = max_size

        counter = Counter()
        for words in word_set:
            for s in words:
                word = self._clean(s)
                counter.update([word])

        d = dict(counter.most_common(max_size - 1))
        print(d)
        d['?'] = max_size

        self._map_from_index = {}
        self._map_to_index = {}

        for i, word in enumerate(d.keys()):
            self._map_to_index[word] = i
            self._map_from_index[i] = word

    def _clean(self, seq):
        s = ''
        for ch in seq.strip():
            if ch.isalpha():
                s += ch

        return s

    def encode(self, s):
        s = self._clean(s)
        if s not in self._map_to_index:
            s = '?'
        return super().encode(s)
