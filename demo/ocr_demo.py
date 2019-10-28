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
from keras.layers import LSTM
from sources.wrappers import OffsetPointsSource, NormalizedSource, Normalizer

from models.ctc_model import CtcModel
from data.h5containers import H5pyDataSet
from sources.wrappers import H5pySource
from factories import BestPathDecodingFactory, TokenPassingDecodingFactory

PORT = 8080
TIMEOUT = 1


def create_source(path):
    return H5pySource(H5pyDataSet(path), random_order=True)


class MyHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        home = DataSetHome('./compiled/ds1', create_source)
        text_encoder = home.get_encoding_table()
        preprocessor = home.get_preprocessor()
        train, val, test = home.get_slices()

        norm_step = preprocessor.steps[1]
        d = norm_step.get_parameters()
        normalizer = Normalizer()
        normalizer.set_mean(d['mu'])
        normalizer.set_deviation(d['sd'])

        if self.path in ['/demo/public/index.html', '/demo/public/index.js']:
            super().do_GET()
        elif self.path == '/get_example':
            source = DenormalizedSource(test, normalizer)
            gen = source.get_sequences()

            x, y = next(gen)

            transcription = ''.join([text_encoder.decode(label) for label in y])

            d = {
                'points': x,
                'transcription': transcription,
                'normalizer': {
                    'muX': normalizer.mu[0],
                    'muY': normalizer.mu[1],
                    'stdX': normalizer.sd[0],
                    'stdY': normalizer.sd[1]
                }
            }

            s = json.dumps(d)
            self.wfile.write(bytes(s, encoding='ascii'))
        elif self.path == '/get_normalizer':
            d = {
                'normalizer': {
                    'muX': normalizer.mu[0],
                    'muY': normalizer.mu[1],
                    'stdX': normalizer.sd[0],
                    'stdY': normalizer.sd[1]
                }
            }

            s = json.dumps(d)
            self.wfile.write(bytes(s, encoding='ascii'))

    def do_POST(self):
        length = int(self.headers.get('content-length'))

        if self.path == '/recognize':
            data_string = self.rfile.read(length)
            points_4d = json.loads(data_string)['line']
            home = DataSetHome('./compiled/ds1', create_source)

            encoding_table = home.get_encoding_table()

            ctc_model = CtcModel(LSTM, 4, encoding_table, 100, './weights/blstm/blstm.h5')

            X = []
            stroke = []
            for x, y, t, eos in points_4d:
                stroke.append((x, y, t))
                if eos == 1:
                    X.append(stroke)
                    stroke = []

            #factory = BestPathDecodingFactory(ctc_model.inference_model, home.get_preprocessor(), home.get_encoding_table())
            factory = TokenPassingDecodingFactory(ctc_model.inference_model, home.get_preprocessor(), home.get_encoding_table())
            predictor = factory.get_predictor()

            s = predictor.predict(X)

            d = {
                'prediction': s
            }

            s = json.dumps(d)
            self.wfile.write(bytes(s, encoding='ascii'))


def open_browser():
    def _open_browser():
        webbrowser.open('http://localhost:{}/{}'.format(PORT, 'demo/public/index.html'))
    thread = threading.Timer(TIMEOUT, _open_browser)
    thread.start()


def start_server():
    server_address = ("", PORT)
    server = HTTPServer(server_address, MyHandler)
    server.serve_forever()


if __name__ == "__main__":
    open_browser()
    start_server()


# todo: fix this demo file
# todo: customize size of canvas
# todo: on end-of-stroke make post request sending points rendered so far to the server and make predictions
# todo: when making predictions, make standard preprocessing, normalization, predict transcription and return it back
# todo: display both predicted labelling and ground-true one
