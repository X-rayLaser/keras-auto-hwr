import unittest
from data.preprocessing import PreProcessor, ProcessingStep, DummyStep
from sources.preloaded import PreLoadedSource
import json


class PreProcessorTests(unittest.TestCase):
    class AddingStep(ProcessingStep):
        def fit(self, data):
            pass

        def process_example(self, x, y):
            return x + 1, y + 1

    class MultiplyingStep(ProcessingStep):
        def fit(self, data):
            pass

        def process_example(self, x, y):
            return x * 2, y * 2

    class DummyStorage:
        def save(self, s):
            self.data = s

        def load(self):
            return self.data

    def test_process_example_with_one_step(self):
        preprocessor = PreProcessor(steps=[self.AddingStep()])
        x, y = preprocessor.pre_process_example(2, 5)
        self.assertEqual(x, 3)
        self.assertEqual(y, 6)

    def test_process_example_with_2_steps(self):
        preprocessor = PreProcessor(steps=[self.AddingStep(),
                                           self.MultiplyingStep()])
        x, y = preprocessor.pre_process_example(2, 5)
        self.assertEqual(x, 6)
        self.assertEqual(y, 12)

    def test_order_of_steps_matters(self):
        preprocessor = PreProcessor(steps=[self.MultiplyingStep(),
                                           self.AddingStep()])
        x, y = preprocessor.pre_process_example(2, 5)
        self.assertEqual(x, 5)
        self.assertEqual(y, 11)

    def test_fit_and_process_example(self):
        preprocessor = PreProcessor(steps=[DummyStep(), DummyStep()])

        xs = [1, 2]
        ys = [0, 0]
        source = PreLoadedSource(xs, ys)
        preprocessor.fit(source)
        x, y = preprocessor.pre_process_example(1, 2)

        self.assertEqual(x, 13)
        self.assertEqual(y, 2)

    def test_save(self):
        step1 = DummyStep()
        step1.s = 3
        step2 = DummyStep()
        step2.s = 9

        preprocessor = PreProcessor(steps=[step1, step2])

        storage = self.DummyStorage()

        preprocessor.save(storage)

        expected = {
            'steps': [{
                    'class_name': 'DummyStep',
                    'params': {'sum': 3}
            }, {
                'class_name': 'DummyStep',
                'params': {'sum': 9}
            }]
        }

        self.assertEqual(storage.data, json.dumps(expected))

    def test_load(self):
        preprocessor = PreProcessor(steps=[])

        storage = self.DummyStorage()

        expected = {
            'steps': [{
                'class_name': 'DummyStep',
                'params': {'sum': 3}
            }, {
                'class_name': 'DummyStep',
                'params': {'sum': 9}
            }]
        }
        storage.data = json.dumps(expected)

        preprocessor.load(storage)

        self.assertEqual(len(preprocessor.steps), 2)
        self.assertIsInstance(preprocessor.steps[0], DummyStep)
        self.assertIsInstance(preprocessor.steps[1], DummyStep)

        self.assertEqual(preprocessor.steps[0].s, 3)
        self.assertEqual(preprocessor.steps[1].s, 9)
