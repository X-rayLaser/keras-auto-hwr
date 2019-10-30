import threading
import webbrowser
from http.server import HTTPServer
from http.server import SimpleHTTPRequestHandler
import json
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from data.data_set_home import DataSetHome
from sources.wrappers import DenormalizedSource, Normalizer

from factories import BestPathDecodingFactory, TokenPassingDecodingFactory
from train_ctc import build_model
from data.data_set_home import create_random_source
PORT = 8080
TIMEOUT = 1


class ResponseStrategy:
    def __init__(self, dataset_location):
        self._home = DataSetHome(dataset_location, create_random_source)

    def normalizer_to_dict(self, normalizer):
        return {
            'muX': normalizer.mu[0],
            'muY': normalizer.mu[1],
            'stdX': normalizer.sd[0],
            'stdY': normalizer.sd[1]
        }

    def get_normalizer(self):
        preprocessor = self._home.get_preprocessor()

        norm_step = preprocessor.steps[1]
        d = norm_step.get_parameters()
        normalizer = Normalizer()
        normalizer.set_mean(d['mu'])
        normalizer.set_deviation(d['sd'])
        return normalizer

    def make_response(self, wfile):
        pass


class GetExample(ResponseStrategy):
    def make_response(self, wfile):
        text_encoder = self._home.get_encoding_table()
        train, val, test = self._home.get_slices()

        normalizer = self.get_normalizer()

        source = DenormalizedSource(test, normalizer)
        gen = source.get_sequences()

        x, y = next(gen)

        transcription = ''.join([text_encoder.decode(label) for label in y])

        s = self.prepare_example(x, transcription, normalizer)
        wfile.write(bytes(s, encoding='ascii'))

    def prepare_example(self, points, transcription, normalizer):
        d = {
            'points': points,
            'transcription': transcription,
            'normalizer': self.normalizer_to_dict(normalizer)
        }

        return json.dumps(d)


class GetNormalizer(ResponseStrategy):
    def make_response(self, wfile):
        normalizer = self.get_normalizer()
        d = {
            'normalizer': self.normalizer_to_dict(normalizer)
        }

        s = json.dumps(d)
        wfile.write(bytes(s, encoding='ascii'))


class MakePrediction(ResponseStrategy):
    def __init__(self, dataset_location, data):
        super().__init__(dataset_location)
        self.data = data

    def make_response(self, wfile):
        data_string = self.data

        points_4d = json.loads(data_string)['line']

        encoding_table = self._home.get_encoding_table()

        ctc_model = build_model(cuda=False, encoding_table=encoding_table)
        model = ctc_model.inference_model
        preprocessor = self._home.get_preprocessor()

        X = self._preprocess_input(points_4d)

        factory = BestPathDecodingFactory(model, preprocessor, encoding_table)
        predictor = factory.get_predictor()

        s = predictor.predict(X)

        d = {
            'prediction': s
        }

        s = json.dumps(d)
        wfile.write(bytes(s, encoding='ascii'))

    def _preprocess_input(self, points_4d):
        X = []
        stroke = []
        for x, y, t, eos in points_4d:
            stroke.append((x, y, t))
            if eos == 1:
                X.append(stroke)
                stroke = []
        return X


def make_handler_class(home_location):
    class MyHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            static_files = ['/demo/public/index.html', '/demo/public/index.js']

            routes = {
                '/get_example': GetExample(home_location),
                '/get_normalizer': GetNormalizer(home_location)
            }

            if self.path in static_files:
                super().do_GET()
            elif self.path in routes:
                response_strategy = routes[self.path]
                response_strategy.make_response(self.wfile)

        def do_POST(self):
            length = int(self.headers.get('content-length'))
            data_string = self.rfile.read(length)

            routes = {
                '/recognize': MakePrediction(home_location, data_string)
            }

            if self.path in routes:
                response_strategy = routes[self.path]
                response_strategy.make_response(self.wfile)

    return MyHandler


def open_browser():
    def _open():
        webbrowser.open('http://localhost:{}/{}'.format(PORT, 'demo/public/index.html'))
    thread = threading.Timer(TIMEOUT, _open)
    thread.start()


def start_server(home_location):
    server_address = ("", PORT)
    handler_class = make_handler_class(home_location)
    server = HTTPServer(server_address, handler_class)
    server.serve_forever()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--home', type=str, default='compiled/ds1')
    parser.add_argument('--token_passing', type=bool, default=False)

    args = parser.parse_args()
    open_browser()
    start_server(args.home)


# todo: customize size of canvas
