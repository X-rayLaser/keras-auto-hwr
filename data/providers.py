from sources.base import BaseSource
from sources.iam_online import OnlineSource, LinesSource
from sources.wrappers import ConstrainedSource, PlainListSource


class DummyProvider(BaseSource):
    def __init__(self, num_lines):
        self.examples = [
            ([[[1]]], '1'), ([[[2]]], '2'), ([[[3]]], '3'), ([[[4]]], '4'), ([[[5]]], '5')
        ]

    def get_sequences(self):
        for x, y in self.examples:
            yield x, y

    def __len__(self):
        return len(self.examples)


class IamSource(BaseSource):
    def __init__(self, num_lines):
        path = './datasets/iam_online_db'
        self._source = ConstrainedSource(
            PlainListSource(LinesSource(OnlineSource(path))), num_lines
        )
        self._num_lines = num_lines

    def get_sequences(self):
        return self._source.get_sequences()

    def __len__(self):
        return self._num_lines
