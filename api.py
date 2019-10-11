import os
from data import providers, preprocessors
from data.data_set_home import DataSetHome
from data.compiler import DataSetCompiler
from data.preprocessing import PreProcessor
from data.factories import DataSplitter
from sources.compiled import H5pyDataSet
from sources.wrappers import H5pySource


class CompilationHome:
    def __init__(self, name):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, 'compiled', name)
        self.root_dir = path
        self.name = name
        self.meta_path = os.path.join(path, 'meta.json')
        self.train_path = os.path.join(path, 'train')
        self.val_path = os.path.join(path, 'validation')
        self.test_path = os.path.join(path, 'test')


class DataRepo:
    def __init__(self, location):
        self._location = location
        self.slices = []
        self._slice_names = ['train', 'validation', 'test']
        self._counter = 0

    def add_slice(self):
        name = self._slice_names[self._counter]
        path = os.path.join(self._location, name)

        ds = H5pyDataSet.create(path)
        self.slices.append(ds)
        self._counter += 1

    def add_example(self, slice_index, x, y):
        self.slices[slice_index].add_example(x, y)


def create_source(path):
    return H5pySource(H5pyDataSet(path), random_order=False)


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

    provider = provider_class(num_examples)
    splitter = DataSplitter.create(provider)
    repo = DataRepo(home.root_dir)
    compiler = DataSetCompiler(preprocessor, splitter, repo)

    compiler.compile()

    DataSetHome.create(providers=[data_provider],
                       preprocessor=preprocessor,
                       slices=repo.slices,
                       create_source=create_source)


def data_set_info(name):
    home = CompilationHome(name)
    with open(home.meta_path, 'r') as f:
        s = f.read()

    return s


def data_set(name):
    home = CompilationHome(name)
    return DataSetHome(home.root_dir, create_source=create_source)


class ProviderNotFoundException(Exception):
    pass


class PreprocessorNotFoundException(Exception):
    pass


# todo: test without split
# todo: test num_lines
