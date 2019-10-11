from sources.base import BaseSource


class DummyProvider(BaseSource):
    def __init__(self):
        self.examples = [
            ([[[1]]], '1'), ([[[2]]], '2'), ([[[3]]], '3'), ([[[4]]], '4'), ([[[5]]], '5')
        ]

    def get_sequences(self):
        for x, y in self.examples:
            yield x, y

    def __len__(self):
        return len(self.examples)
