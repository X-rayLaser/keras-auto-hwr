from unittest import TestCase


class PreProcessorMock:
    def __init__(self, output):
        self.counter = 0
        self.args = None
        self.output = output

    def pre_process(self, *args):
        self.counter += 1
        self.args = args
        return self.output


class InferenceModelMock:
    def __init__(self):
        self.predict_called = False
        self.args = None

    def predict(self, *args):
        self.predict_called = True
        self.args = args


class InputAdapterMock:
    def __init__(self):
        self.called = False
        self.args = None

    def adapt_input(self, *args):
        self.called = True
        self.args = args


class Predictor:
    def __init__(self, model, preprocessor, input_adapter):
        self.model = model
        self.preprocessor = preprocessor
        self.input_adapter = input_adapter

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
        x = self.preprocessor.pre_process(strokes)
        self.input_adapter.adapt_input(x)
        self.model.predict()


class UnexpectedInputException(Exception):
    pass


class ZeroLengthInputException(Exception):
    pass


class PredictorTests(TestCase):
    def test_predict_with_invalid_input(self):
        model = InferenceModelMock()
        preprocessor = PreProcessorMock([])
        input_adapter = InputAdapterMock()
        predictor = Predictor(model, preprocessor, input_adapter)

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

        adapted_x = []

        model = InferenceModelMock()
        preprocessor = PreProcessorMock(processed_strokes)
        input_adapter = InputAdapterMock()
        predictor = Predictor(model, preprocessor, input_adapter)

        predictor.predict(strokes)

        self.assertEqual(preprocessor.counter, 1)
        self.assertEqual(preprocessor.counter, 1)

        self.assertEqual(preprocessor.args, (strokes, ))

        self.assertTrue(input_adapter.called)

        self.assertEqual(input_adapter.args, (processed_strokes,))

        self.assertTrue(model.predict_called)

        #self.assertEqual(model.args, (processed_strokes,))


# todo: check model.predict parameters and output
# todo: model.predict should return a list of class labels
# todo: additional object is responsible for parsing those classes and turning them to final strings
