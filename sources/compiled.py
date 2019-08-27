import json

from sources import BaseSource
from sources.iam_online import Stroke


class CompilationSource(BaseSource):
    def __init__(self, path):
        self._path = path

        with open(self._path, 'r') as f:
            s = f.read()
        d = json.loads(s)

        self._hand_writings = d['hand_writings']
        self._transcriptions = d['transcriptions']

    def get_sequences(self):
        for i in range(len(self._transcriptions)):
            strokes = []
            hand_writing = self._hand_writings[i]
            for stroke_points in hand_writing:
                strokes.append(Stroke(stroke_points))

            yield strokes, self._transcriptions[i]

    def __len__(self):
        return len(self._transcriptions)
