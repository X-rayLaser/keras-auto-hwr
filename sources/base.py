class BaseSource:
    def get_sequences(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
