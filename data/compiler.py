from sources.wrappers import PreprocessedSource


class DataSetCompiler:
    def __init__(self, preprocessor, splitter, repo):
        self._preprocessor = preprocessor
        self._splitter = splitter
        self._repo = repo

    def compile(self):
        self._splitter.split()
        self._preprocessor.fit(self._splitter.train_data())

        train_source = PreprocessedSource(self._splitter.train_data(),
                                          self._preprocessor)
        val_source = PreprocessedSource(self._splitter.validation_data(),
                                        self._preprocessor)

        test_source = PreprocessedSource(self._splitter.test_data(),
                                         self._preprocessor)

        self._repo.add_slice()
        self._repo.add_slice()
        self._repo.add_slice()

        for xs, ys in train_source.get_sequences():
            self._repo.add_example(0, xs, ys)

        for xs, ys in val_source.get_sequences():
            self._repo.add_example(1, xs, ys)

        for xs, ys in test_source.get_sequences():
            self._repo.add_example(2, xs, ys)
