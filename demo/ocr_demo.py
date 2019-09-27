import threading
import webbrowser
from http.server import HTTPServer
from http.server import SimpleHTTPRequestHandler
import json
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from sources.compiled import CompilationSource
from sources.wrappers import DenormalizedSource, Normalizer


PORT = 8080
TIMEOUT = 1


class MyHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ['/demo/public/index.html', '/demo/public/index.js']:
            super().do_GET()
        elif self.path == '/get_example':

            source = CompilationSource('./compiled/test.h5py', num_lines=10)
            normalizer = Normalizer.from_json('./compiled/mu_std.json')
            source = DenormalizedSource(source, normalizer)
            gen = source.get_sequences()

            x, y = next(gen)

            d = {
                'points': x,
                'transcription': y,
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
        data_string = self.rfile.read(length)
        x = json.loads(data_string)['number']
        try:
            result = int(x) ** 2
        except:
            import traceback
            s = traceback.format_exc()
            result = s

        d = {
            'x': [[result], [result], [result]],
            'y': 'Hahah'
        }
        s = json.dumps(result)
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


# todo: customize size of canvas
# todo: fetch example, render points using setTimeout, transform points if necessary to fit them on canvas
# todo: on end-of-stroke make post request sending points rendered so far to the server and make predictions
# todo: when making predictions, make standard preprocessing, normalization, predict transcription and return it back
# todo: display both predicted labelling and ground-true one
