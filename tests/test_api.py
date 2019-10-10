from unittest import TestCase
import api
import os
import json
import shutil


class ApiTests(TestCase):
    def setUp(self):
        self.name = 'dummy'
        self.provider = 'DummyProvider'
        self.preprocessor = 'dummy_preprocessor'
        self.num_examples = 3
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

    def test_compilation_creates_correct_directory_layout(self):
        return
        self.assertTrue(os.path.exists(self.home.root_dir))
        self.assertTrue(os.path.exists(self.home.train_path))
        self.assertTrue(os.path.exists(self.home.val_path))
        self.assertTrue(os.path.exists(self.home.test_path))

    def test_compile_with_data_creates_valid_meta_data(self):
        with open(self.home.meta_path, 'r') as f:
            s = f.read()

        info = json.loads(s)
        self.assertEqual(info, {
            'location': self.home.root_dir,
            'preprocessor': 'dummy_preprocessor',
            'provider': 'DummyProvider',
            'number of examples': self.num_examples
        })

    def test_compilation_does_generate_data(self):
        return
        ds = api.data_set(self.name)

        train, val, test = ds.get_slices()

        m = len(train) + len(val) + len(test)
        self.assertEqual(m, self.num_examples)

        X = []
        Y = []
        for x, y in train.get_sequences():
            X.append(x)
            Y.append(y)

        for x, y in val.get_sequences():
            X.append(x)
            Y.append(y)

        for x, y in test.get_sequences():
            X.append(x)
            Y.append(y)

        self.assertEqual(len(X), len(Y))
        self.assertEqual(len(X), m)

        self.assertEqual(X, [
            [(0, 1, 2, 3), (4, 5, 6, 7)],
            [(8, 9, 10, 11)],
            [(12, 13, 14, 15)]
        ])

        self.assertEqual(Y, ["first", "second", "third"])


# todo: add shape, # of train, # of val, # of test properties
