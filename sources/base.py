class BaseSource:
    def get_sequences(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class BaseSourceWrapper(BaseSource):
    def __init__(self, source):
        self._source = source

    def __len__(self):
        return len(self._source)
