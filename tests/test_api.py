from unittest import TestCase
import api
import os
import shutil
from data.providers import DummyProvider


class ApiTests(TestCase):
    def setUp(self):
        self.name = 'dummy'
        self.provider = 'DummyProvider'
        self.preprocessor = 'dummy_preprocessor'
        self.num_examples = 5
        self.home = api.CompilationHome(self.name)

        api.compile_data_set(data_provider=self.provider,
                             preprocessor_name=self.preprocessor,
                             name=self.name, num_examples=self.num_examples)

    def tearDown(self):
        if os.path.exists(self.home.root_dir):
            shutil.rmtree(self.home.root_dir)

    def test_compile_with_invalid_parameters(self):
        self.assertRaises(
            api.ProviderNotFoundException,
            lambda: api.compile_data_set(
                data_provider="foo", preprocessor_name=self.preprocessor,
                name=self.name, num_examples=self.num_examples)
        )

        self.assertRaises(
            api.PreprocessorNotFoundException,
            lambda: api.compile_data_set(
                data_provider=self.provider, preprocessor_name="foo",
                name=self.name, num_examples=self.num_examples)
        )

    def test_compilation_does_generate_data(self):
        ds = api.data_set(self.name)

        train, val, test = ds.get_slices()
        m = len(train) + len(val) + len(test)
        self.assertEqual(m, self.num_examples)

        res = list(train.get_sequences()) + list(val.get_sequences())\
              + list(test.get_sequences())

        self.assertEqual(len(res), m)

        import numpy as np
        expected = [(np.array(x) + 1, y) for x, y in DummyProvider().examples]

        self.assertEqual(res, expected)


# todo: add shape, # of train, # of val, # of test properties
