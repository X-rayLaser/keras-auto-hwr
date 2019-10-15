from data.preprocessing import ProcessingStep, OffsetStep, NormalizationStep,\
    LabelEncodingStep, WordEncodingStep


class DummyPreprocessor:
    pass


class AddOne(ProcessingStep):
    def process_x(self, x):
        val = x[0][0][0] + 1
        return [[[val]]]


dummy_preprocessor = [(AddOne, {})]

default = [(OffsetStep, {}), (NormalizationStep, {}), (LabelEncodingStep, {})]

word_level = [(OffsetStep, {}), (NormalizationStep, {}), (WordEncodingStep, {'num_words': 500})]
