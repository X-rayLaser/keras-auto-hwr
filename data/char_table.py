import json


class CharacterTable:
    def __init__(self):
        index_to_char = {}
        char_to_index = {}

        first_char = 32
        last_char = 127
        for code in range(first_char, last_char + 1):
            ch = chr(code)
            index = code - first_char
            index_to_char[index] = ch
            char_to_index[ch] = index

        sentinel_index = len(index_to_char)
        index_to_char[sentinel_index] = self.sentinel
        char_to_index[self.sentinel] = sentinel_index

        self._map_to_index = char_to_index
        self._map_from_index = index_to_char

    def save(self, path):
        d = {
            'map_to_index': self._map_to_index,
            'map_from_index': self._map_from_index
        }
        json_str = json.dumps(d)
        with open(path, 'w') as f:
            f.write(json_str)

    def load(self, path):
        with open(path, 'r') as f:
            s = f.read()

        d = json.loads(s)
        self._map_to_index = d['map_to_index']
        self._map_from_index = d['map_from_index']

        d = {}
        for k, v in self._map_from_index.items():
            d[int(k)] = v

        self._map_from_index = d

    def is_sentinel(self, ch):
        return self.sentinel == ch

    def is_unknown(self, index):
        return index not in self._map_from_index

    @property
    def sentinel(self):
        return '\n'

    @property
    def start(self):
        return '_'

    def encode(self, ch):
        return self._map_to_index[ch]

    def decode(self, index):
        if index in self._map_from_index:
            return self._map_from_index[index]
        else:
            return '?'

    def __len__(self):
        return len(self._map_to_index)
