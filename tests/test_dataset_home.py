from unittest import TestCase
import json

from data.data_set_home import DataSetHome
from data.preprocessing import PreProcessor
from data import preprocessing
import os
from sources.base import BaseSource
import shutil


class FileSourceMock(BaseSource):
    def __init__(self, path):
        with open(path, 'r') as f:
            s = f.read()

        d = json.loads(s)
        self.examples = d['examples']
        self._path = path

    def get_sequences(self):
        for x, y in self.examples:
            yield x, y

    @property
    def location(self):
        return self._path

    def __len__(self):
        return len(list(self.get_sequences()))


class DataSetHomeLoadingTests(TestCase):
    def _expected_dict(self):
        examples_total = len(self.train_examples)\
                         + len(self.validation_examples)\
                         + len(self.test_examples)
        encoding_step = preprocessing.EnglishEncodingStep()
        return {
            'location_dir': self.home_dir,
            'providers': ['DummyProvider'],
            'preprocessor_steps': [
                {'class_name': 'DummyStep', 'params': {'sum': 1}},
                {'class_name': 'DummyStep', 'params': {'sum': 12}},
                {
                    'class_name': 'EnglishEncodingStep',
                    'params': encoding_step.get_parameters()
                },
            ],
            'number of examples': examples_total,
            'slices': {
                'train': len(self.train_examples),
                'validation': len(self.validation_examples),
                'test': len(self.test_examples)
            }
        }

    def populate_file(self, path, examples):
        with open(path, 'w') as f:
            f.write(json.dumps({'examples': examples}))

    def setUp(self):
        self.home_dir = './data_set_home'
        self.train_path = os.path.join(self.home_dir, 'train')
        self.validation_path = os.path.join(self.home_dir, 'validation')
        self.test_path = os.path.join(self.home_dir, 'test')

        self.train_examples = [
            (1, '1'), (2, '2'), (3, '3'), (4, '4'), (5, '5'), (6, '6'), (7, '7')
        ]

        self.validation_examples = [(8, '8'), (9, '9')]

        self.test_examples = [(10, '10')]

        os.makedirs(self.home_dir)
        self.populate_file(self.train_path, self.train_examples)
        self.populate_file(self.validation_path, self.validation_examples)
        self.populate_file(self.test_path, self.test_examples)

        self.expected_dict = self._expected_dict()

        data = json.dumps(self.expected_dict)

        meta_path = os.path.join(self.home_dir, 'meta.json')
        with open(meta_path, 'w') as f:
            f.write(data)

        self.home = DataSetHome(self.home_dir, lambda path: FileSourceMock(path))

    def tearDown(self):
        if os.path.isdir(self.home_dir):
            shutil.rmtree(self.home_dir)

    def test_load_meta_info(self):
        expected_dict = self.expected_dict
        self.assertEqual(self.home.meta_info, expected_dict)

    def test_get_preprocessor(self):
        preprocessor = self.home.get_preprocessor()

        self.assertIsInstance(preprocessor, PreProcessor)
        self.assertEqual(len(preprocessor.steps), 3)

        self.assertEqual(preprocessor.steps[0].get_parameters(), {
            'sum': 1
        })

        self.assertEqual(preprocessor.steps[1].get_parameters(), {
            'sum': 12
        })

        encoding_step = preprocessing.EnglishEncodingStep()

        self.assertEqual(preprocessor.steps[2].get_parameters(),
                         encoding_step.get_parameters())

    def test_get_encoding_table(self):
        encoding_table = self.home.get_encoding_table()
        ch = 'S'
        self.assertGreaterEqual(len(encoding_table), 26)
        self.assertEqual(encoding_table.decode(encoding_table.encode(ch)), ch)

    def test_get_slices(self):
        slices = self.home.get_slices()
        self.assertEqual(len(slices), 3)

        self.assertEqual(list(slices[0].get_sequences()), self.train_examples)
        self.assertEqual(list(slices[1].get_sequences()),
                         self.validation_examples)
        self.assertEqual(list(slices[2].get_sequences()),
                         self.test_examples)

    def test_generates_correct_meta_info_object(self):
        providers = ['DummyProvider']

        step1 = preprocessing.DummyStep()
        step2 = preprocessing.DummyStep()
        step3 = preprocessing.EnglishEncodingStep()

        step1.set_parameters({'sum': 1})
        step2.set_parameters({'sum': 12})

        preprocessor = PreProcessor([step1, step2, step3])

        data_slices = [FileSourceMock(self.train_path),
                       FileSourceMock(self.validation_path),
                       FileSourceMock(self.test_path)]
        home = DataSetHome.create(providers, preprocessor,
                                  data_slices, lambda path: FileSourceMock(path))

        self.assertEqual(home.meta_info, self.expected_dict)
