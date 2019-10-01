from sources.base import BaseSourceWrapper
from sources.wrappers import Normalizer
import json
import sys


class ProcessingStep:
    def set_parameters(self, params_dict):
        pass

    def fit(self, data):
        pass

    def process_example(self, x, y):
        raise NotImplementedError

    def get_parameters(self):
        return {}


class OffsetStep(ProcessingStep):
    def _pre_process(self, x, x0, y0, t0):
        new_seq = []
        for stroke in x:
            points = []

            for x, y, t in stroke.points:
                points.append((x - x0, y - y0, t - t0, 0))

            points[-1] = points[-1][:-1] + (1,)
            new_seq.extend(points)
        return new_seq

    def process_example(self, x, y):
        x0, y0, t0 = x.points[0]
        new_seq = self._pre_process(x, x0, y0, t0)
        return new_seq, y

    def set_parameters(self, params_dict):
        pass


class NormalizationStep(ProcessingStep):
    def __init__(self):
        self._normalizer = Normalizer()

    def fit(self, data):
        xs = (in_seq for in_seq, _ in data)
        self._normalizer.fit(xs)

    def process_example(self, x, y):
        norm = self._normalizer.preprocess([x])[0]
        return norm, y

    def set_parameters(self, params_dict):
        mu = params_dict['mu']
        sd = params_dict['sd']
        self._normalizer.set_mean(mu)
        self._normalizer.set_deviation(sd)

    def get_parameters(self):
        return {
            'mu': self._normalizer.mu,
            'sd': self._normalizer.sd
        }


class DummyStep(ProcessingStep):
    def fit(self, data):
        self.s = sum((xs for xs, ys in data))

    def process_example(self, x, y):
        return x + self.s, y

    def set_parameters(self, params_dict):
        self.s = params_dict['sum']

    def get_parameters(self):
        return {'sum': self.s}


class TextStorage:
    def __init__(self, path):
        self._path = path

    def save(self, s):
        with open(self._path, 'w') as f:
            f.write(s)

    def load(self):
        with open(self._path, 'r') as f:
            return f.read()


class PreProcessor:
    def __init__(self, steps):
        self._steps = steps

    @property
    def steps(self):
        return self._steps

    def fit(self, source):
        gen = ((xs, ys) for xs, ys in source.get_sequences())

        for i, step in enumerate(self._steps):
            step.fit(gen)
            stop_index = i + 1
            gen = (self._preprocess(xs, ys, stop_index) for xs, ys in source.get_sequences())

    def _preprocess(self, x, y, stop_index):
        for i, step in enumerate(self._steps):
            if i >= stop_index:
                break

            x, y = step.process_example(x, y)

        return x, y

    def pre_process_example(self, x, y):
        return self._preprocess(x, y, stop_index=len(self._steps))

    def save(self, storage):
        steps = []
        for step in self._steps:
            steps.append({
                'class_name': step.__class__.__name__,
                'params': step.get_parameters()
            })
        d = {'steps': steps}

        s = json.dumps(d)
        storage.save(s)

    def load(self, storage):
        s = storage.load()

        d = json.loads(s)

        self._steps = []
        this_module = sys.modules[__name__]
        for info in d['steps']:
            cls_name = info['class_name']
            params = info['params']

            cls = getattr(this_module, cls_name)

            step = cls()
            step.set_parameters(params)
            self._steps.append(step)
