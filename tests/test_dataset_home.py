from unittest import TestCase
import json
from data.preprocessing import PreProcessor
from data import preprocessing
import os
from sources.base import BaseSource
import shutil


class DataSetHome:
    def __init__(self, location):
        self._location = location

    @property
    def meta_info(self):
        path = os.path.join(self._location, 'meta.json')
        with open(path, 'r') as f:
            s = f.read()
        d = json.loads(s)
        return d

    def get_preprocessor(self):
        steps = []
        for step_dict in self.meta_info['preprocessor_steps']:
            cls = getattr(preprocessing, step_dict['class_name'])
            params = step_dict['params']
            step = cls()
            step.set_parameters(params)
            steps.append(step)

        return PreProcessor(steps)

    def slice_path(self, slice_name):
        return os.path.join(self.meta_info['location_dir'], slice_name)

    def get_slices(self):
        slices = []
        for k, v in self.meta_info['slices'].items():
            path = self.slice_path(k)
            source = FileSourceMock(path)
            slices.append(source)
        return slices

    @staticmethod
    def create(providers, preprocessor, slices):
        location_dir = ''
        slices_dict = {}

        examples_total = 0
        for data_slice in slices:
            location_dir, file_name = os.path.split(data_slice.location)
            num_examples = len(data_slice)
            slices_dict[file_name] = num_examples
            examples_total += num_examples

        steps = []
        for step in preprocessor.steps:
            steps.append({
                'class_name': step.__class__.__name__,
                'params': step.get_parameters()
            })

        d = {
            'location_dir': location_dir,
            'providers': providers,
            'preprocessor_steps': steps,
            'number of examples': examples_total,
            'slices': slices_dict
        }

        path = os.path.join(location_dir, 'meta.json')
        with open(path, 'w') as f:
            data = json.dumps(d)
            f.write(data)

        return DataSetHome(location_dir)


class Storage:
    def __init__(self, data):
        self.data = data


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
        return {
            'location_dir': self.home_dir,
            'providers': ['DummyProvider'],
            'preprocessor_steps': [
                {'class_name': 'DummyStep', 'params': {'sum': 1}},
                {'class_name': 'DummyStep', 'params': {'sum': 12}},
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

        self.home = DataSetHome(self.home_dir)

    def tearDown(self):
        if os.path.isdir(self.home_dir):
            shutil.rmtree(self.home_dir)

    def test_load_meta_info(self):
        expected_dict = self.expected_dict
        self.assertEqual(self.home.meta_info, expected_dict)

    def test_get_preprocessor(self):
        preprocessor = self.home.get_preprocessor()

        self.assertIsInstance(preprocessor, PreProcessor)
        self.assertEqual(len(preprocessor.steps), 2)

        self.assertEqual(preprocessor.steps[0].get_parameters(), {
            'sum': 1
        })

        self.assertEqual(preprocessor.steps[1].get_parameters(), {
            'sum': 12
        })

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

        step1.set_parameters({'sum': 1})
        step2.set_parameters({'sum': 12})

        preprocessor = PreProcessor([step1, step2])

        data_slices = [FileSourceMock(self.train_path),
                       FileSourceMock(self.validation_path),
                       FileSourceMock(self.test_path)]
        home = DataSetHome.create(providers, preprocessor, data_slices)

        self.assertEqual(home.meta_info, self.expected_dict)
