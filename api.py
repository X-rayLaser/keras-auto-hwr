import json
import os
from data import providers, preprocessors


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

    from data.compiler import DataSetCompiler
    from data.preprocessing import PreProcessor

    #steps = [step_cls(**params) for step_cls, params in preprocessor_steps]

    #preprocessor = PreProcessor(steps)

    provider = provider_class()
    #DataSetCompiler(provider, preprocessor, repo)

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
    return DataSet()


class ProviderNotFoundException(Exception):
    pass


class PreprocessorNotFoundException(Exception):
    pass
