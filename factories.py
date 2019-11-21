from data.example_adapters import CTCAdapter
from models.ctc_model import CTCOutputDecoder, BestPathClassPredictor, TokenPassingPredictor
from tests.test_predictor import Predictor
from data.language_models import WordDictionary


class PredictorFactory:
    def get_predictor(self):
        raise NotImplementedError


class CTCPredictorFactory(PredictorFactory):
    def __init__(self, model, preprocessor, text_encoder):
        self._model = model
        self._preprocessor = preprocessor
        self._text_encoder = text_encoder

    def get_adapter(self, padding):
        return CTCAdapter(padding)


class BestPathDecodingFactory(CTCPredictorFactory):
    def get_predictor(self):
        adapter = self.get_adapter(self._text_encoder.sentinel)
        decoder = CTCOutputDecoder(self._text_encoder)

        class_predictor = BestPathClassPredictor(self._model)
        return Predictor(class_predictor, self._preprocessor, adapter, decoder)


class TokenPassingDecodingFactory(CTCPredictorFactory):
    class OutputDecoder:
        def __init__(self, word_dict, text_encoder):
            self._word_dict = word_dict
            self._text_encoder = text_encoder

        def decode(self, labels):
            words = [self._word_dict.words[label] for label in labels]
            return ' '.join(words)

    def get_predictor(self):
        adapter = self.get_adapter(self._text_encoder.sentinel)

        word_dict = WordDictionary.load('english.json')
        class_predictor = TokenPassingPredictor(self._model, word_dict, self._text_encoder)

        decoder = self.OutputDecoder(word_dict, self._text_encoder)

        return Predictor(class_predictor, self._preprocessor, adapter, decoder)
