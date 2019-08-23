import json


class CompilationSource:
    def __init__(self, path):
        self._path = path

        with open(self._path, 'r') as f:
            s = f.read()
        d = json.loads(s)

        self._hand_writings = d['hand_writings']
        self._transcriptions = d['transcriptions']

    def get_lines(self):

        for i in range(len(self._transcriptions)):
            yield self._hand_writings[i], self._transcriptions[i]

    def __len__(self):
        return len(self._transcriptions)
