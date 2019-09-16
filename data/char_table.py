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

        self._char_to_index = char_to_index
        self._index_to_char = index_to_char

    def save(self, path):
        import json
        d = {
            'char_to_index': self._char_to_index,
            'index_to_char': self._index_to_char
        }
        json_str = json.dumps(d)
        with open(path, 'w') as f:
            f.write(json_str)

    def load(self, path):
        import json
        with open(path, 'r') as f:
            s = f.read()

        d = json.loads(s)
        self._char_to_index = d['char_to_index']
        self._index_to_char = d['index_to_char']

        d = {}
        for k, v in self._index_to_char.items():
            d[int(k)] = v

        self._index_to_char = d

    def is_sentinel(self, ch):
        return self.sentinel == ch

    @property
    def sentinel(self):
        return '\n'

    @property
    def start(self):
        return '_'

    def encode(self, ch):
        return self._char_to_index[ch]

    def decode(self, index):
        if index in self._index_to_char:
            return self._index_to_char[index]
        else:
            raise Exception('Unknown {}'.format(index))
            return '?'

    def __len__(self):
        return len(self._char_to_index)