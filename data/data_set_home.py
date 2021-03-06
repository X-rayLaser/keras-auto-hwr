import json
import os
from data import preprocessing, PreProcessor
from data.encodings import TextEncodingTable
from sources.wrappers import H5pySource
from data.h5containers import H5pyDataSet
from data.preprocessing import Normalizer
from data import providers


def create_deterministic_source(path):
    return H5pySource(H5pyDataSet(path), random_order=False)


def create_random_source(path):
    return H5pySource(H5pyDataSet(path), random_order=True)


class DataSetHome:
    def __init__(self, location, create_source):
        self._location = location
        self._create_source = create_source

    @property
    def meta_info(self):
        path = os.path.join(self._location, 'meta.json')
        with open(path, 'r') as f:
            s = f.read()
        d = json.loads(s)
        return d

    def get_encoding_table(self):
        last_step = self.get_preprocessor().steps[-1]
        mapping_dict = last_step.get_parameters()
        return TextEncodingTable(mapping_dict)

    def get_preprocessor(self):
        steps = []
        for step_dict in self.meta_info['preprocessor_steps']:
            cls = getattr(preprocessing, step_dict['class_name'])
            params = step_dict['params']
            step = cls()
            step.set_parameters(params)
            steps.append(step)

        return PreProcessor(steps)

    def get_normalizer(self):
        preprocessor = self.get_preprocessor()

        norm_step = preprocessor.steps[1]
        d = norm_step.get_parameters()
        normalizer = Normalizer()
        normalizer.set_mean(d['mu'])
        normalizer.set_deviation(d['sd'])
        return normalizer

    def get_provider(self):
        provider_cls_name = self.meta_info['providers'][0]
        cls = getattr(providers, provider_cls_name)
        return cls(10000)

    def slice_path(self, slice_name):
        return os.path.join(self.meta_info['location_dir'], slice_name)

    def get_slices(self):
        slices = []
        for k, v in self.meta_info['slices'].items():
            path = self.slice_path(k)
            source = self._create_source(path)
            slices.append(source)
        return slices

    @staticmethod
    def create(providers, preprocessor, slices, create_source):
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

        return DataSetHome(location_dir, create_source)
