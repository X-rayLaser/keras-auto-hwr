import json
import os
from data import providers, preprocessors
from tests.test_dataset_home import DataSetHome
from tests.test_dataset_compiler import DataSetCompiler
from data.preprocessing import PreProcessor
from data.factories import DataSplitter
from sources.compiled import H5pyDataSet


class CompilationHome:
    def __init__(self, name):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, 'compiled', name)
        self.root_dir = path
        self.name = name
        self.meta_path = os.path.join(path, 'data_set.json')
        self.train_path = os.path.join(path, 'train')
        self.val_path = os.path.join(path, 'validation')
        self.test_path = os.path.join(path, 'test')


class DataRepoMock:
    def __init__(self, location):
        self._location = location
        self.slices = []
        self._slice_names = ['train', 'validation', 'test']
        self._counter = 0

    def add_slice(self):
        print('jfliaejfliajelifjeifjlij')

        name = self._slice_names[self._counter]
        path = os.path.join(self._location, name)

        ds = H5pyDataSet.create(path)
        self.slices.append(ds)
        self._counter += 1

    def add_example(self, slice_index, x, y):
        print('EHAILEFJLEIJFIJ')
        self.slices[slice_index].append((x, y))


def compile_data_set(data_provider, preprocessor_name, name, num_examples):
    try:
        provider_class = getattr(providers, data_provider)
    except AttributeError:
        raise ProviderNotFoundException()

    try:
        preprocessor_steps = getattr(preprocessors, preprocessor_name)
    except AttributeError:
        raise PreprocessorNotFoundException()

    home = CompilationHome(name)

    if not os.path.isdir(home.root_dir):
        os.makedirs(home.root_dir)

    steps = [step_cls(**params) for step_cls, params in preprocessor_steps]

    preprocessor = PreProcessor(steps)

    provider = provider_class()
    #splitter = DataSplitter(provider)
    repo = DataRepoMock(home.root_dir)
    #compiler = DataSetCompiler(provider, preprocessor, splitter, repo)

    #compiler.compile()

    data_set_home = DataSetHome.create(providers=[data_provider],
                                       preprocessor=preprocessor,
                                       slices=repo.slices)

    d = {
        'location': home.root_dir,
        'preprocessor': preprocessor_name,
        'provider': data_provider,
        'number of examples': num_examples
    }

    s = json.dumps(d)
    with open(home.meta_path, 'w') as f:
        f.write(s)


def data_set_info(name):
    home = CompilationHome(name)
    with open(home.meta_path, 'r') as f:
        s = f.read()

    return s


class DataSet:
    @property
    def partitions(self):
        return 0, 0, 0


def data_set(name):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    location = os.path.join(current_dir, name)
    return DataSetHome(location)


class ProviderNotFoundException(Exception):
    pass


class PreprocessorNotFoundException(Exception):
    pass
