import json

from sources import BaseSource


class CompilationSource(BaseSource):
    def __init__(self, path, num_lines):
        self._path = path
        self._num_lines = num_lines

        with open(self._path, 'r') as f:
            s = f.read()
        d = json.loads(s)

        self._hand_writings = d['hand_writings']
        self._transcriptions = d['transcriptions']

    def get_sequences(self):
        for i in range(len(self)):
            yield self._hand_writings[i], self._transcriptions[i]

    def __len__(self):
        return min(self._num_lines, len(self._transcriptions))
