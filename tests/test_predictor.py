from unittest import TestCase
import numpy as np


class PreProcessorMock:
    def __init__(self, output):
        self.args = None
        self.output = output

    def pre_process(self, *args):
        self.args = args
        return self.output


class InferenceModelMock:
    def __init__(self, output):
        self.args = None
        self.output = output

    def predict(self, *args):
        self.args = args
        return self.output


class InputAdapterMock:
    def __init__(self, output):
        self.args = None
        self.output = output

    def adapt_input(self, *args):
        self.args = args
        return self.output


class OutputDecoderMock:
    def __init__(self, output):
        self.args = None
        self.output = output

    def decode(self, *args):
        self.args = args
        return self.output


class Predictor:
    def __init__(self, model, preprocessor, input_adapter, output_decoder):
        self.model = model
        self.preprocessor = preprocessor
        self.input_adapter = input_adapter
        self.decoder = output_decoder

    def _validate_list_of_lists(self, strokes):
        class_name = type(strokes).__name__
        msg = 'Expected list of lists of tuples'
        if class_name != 'list':
            raise UnexpectedInputException(msg)

        if len(strokes) == 0:
            raise ZeroLengthInputException()

        for stroke in strokes:
            if type(stroke).__name__ != 'list':
                raise UnexpectedInputException(msg)

    def _first_tuple_size(self, strokes):
        for stroke in strokes:
            for point in stroke:
                return len(point)

    def _validate_points(self, strokes):
        for stroke in strokes:
            for point in stroke:
                if type(point).__name__ not in ['list', 'tuple']:
                    raise UnexpectedInputException('')

    def _validate(self, strokes):
        self._validate_list_of_lists(strokes)
        self._validate_points(strokes)

        tuple_len = self._first_tuple_size(strokes)

        if tuple_len is None:
            return

        if tuple_len == 0:
            raise UnexpectedInputException(
                'Tuples must contain at least 1 element'
            )

        for stroke in strokes:
            for point in stroke:
                if len(point) != tuple_len:
                    raise UnexpectedInputException(
                        'All tuples must be of the same length'
                    )

    def predict(self, strokes):
        self._validate(strokes)
        processed = self.preprocessor.pre_process(strokes)
        adapted = self.input_adapter.adapt_input(processed)
        classes = self.model.predict(adapted)
        return self.decoder.decode(classes)


class UnexpectedInputException(Exception):
    pass


class ZeroLengthInputException(Exception):
    pass


class PredictorTests(TestCase):
    def test_predict_with_invalid_input(self):
        model = InferenceModelMock([])
        preprocessor = PreProcessorMock([])
        input_adapter = InputAdapterMock([])
        output_decoder = OutputDecoderMock('')

        predictor = Predictor(model, preprocessor, input_adapter, output_decoder)

        self.assertRaises(UnexpectedInputException, lambda: predictor.predict(4))

        self.assertRaises(ZeroLengthInputException, lambda: predictor.predict([]))
        self.assertRaises(UnexpectedInputException, lambda: predictor.predict([32, [(4, 5)]]))
        self.assertRaises(UnexpectedInputException, lambda: predictor.predict([
            [4, 2], [3, 2, 2]
        ]))

        self.assertRaises(UnexpectedInputException, lambda: predictor.predict([
            [(4, 2), (3, 2)], [(0, 1, 3), (2, 5, 8, 15)]
        ]))

    def test_predict(self):
        strokes = [
            [(0, 0), (1, 2)], [(3, 3)]
        ]

        processed_strokes = [(1, 2), (3, 3)]

        adapted_x = np.array(processed_strokes).reshape((1, 2, 2))

        classes = [0, 2]

        output = 'hello, world'

        model = InferenceModelMock(classes)
        preprocessor = PreProcessorMock(processed_strokes)
        input_adapter = InputAdapterMock(adapted_x)
        output_decoder = OutputDecoderMock(output)
        predictor = Predictor(model, preprocessor, input_adapter, output_decoder)

        res = predictor.predict(strokes)

        self.assertEqual(preprocessor.args, (strokes, ))

        self.assertEqual(input_adapter.args, (processed_strokes,))

        self.assertEqual(model.args, (adapted_x,))

        self.assertEqual(output_decoder.args, (classes,))

        self.assertEqual(res, output)


# todo: check model.predict parameters and output
# todo: model.predict should return a list of class labels
# todo: additional object is responsible for parsing those classes and turning them to final strings
