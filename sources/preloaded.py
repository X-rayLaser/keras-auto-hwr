import random

from sources import BaseSource


class PreLoadedSource(BaseSource):
    def __init__(self, hand_writings, transcriptions):
        self._num_examples = len(transcriptions)
        self._line_points = hand_writings
        self._transcriptions = transcriptions

    def __len__(self):
        return len(self._transcriptions)

    def get_sequences(self):
        indices = list(range(len(self._transcriptions)))

        random.shuffle(indices)

        for i in indices:
            yield self._line_points[i], self._transcriptions[i]