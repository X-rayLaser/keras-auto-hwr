from data.preprocessing import ProcessingStep


class DummyPreprocessor:
    pass


class AddOne(ProcessingStep):
    def process_x(self, x):
        return x + 1


dummy_preprocessor = [(AddOne, {})]
