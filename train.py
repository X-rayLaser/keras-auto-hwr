import xml.etree.ElementTree as ET
from PIL.ImageDraw import ImageDraw
from PIL import Image
import numpy as np
import os
from data import Transcription, StrokeLine


class CharacterTable:
    def __init__(self, corpus):
        charset = set(corpus + self.sentinel + self.start)
        self._char_to_index = dict((ch, index) for index, ch in enumerate(charset))
        self._index_to_char = dict((v, k) for k, v in self._char_to_index.items())

    def save(self, path):
        import json
        d = {
            'char_to_index': self._char_to_index,
            'index_to_char': self._index_to_char
        }
        json_str = json.dumps(d)
        with open(path, 'w') as f:
            f.write(json_str)

    def load(self, path):
        import json
        with open(path, 'r') as f:
            s = f.read()

        d = json.loads(s)
        self._char_to_index = d['char_to_index']
        self._index_to_char = d['index_to_char']

        d = {}
        for k, v in self._index_to_char.items():
            d[int(k)] = v

        self._index_to_char = d

    def is_sentinel(self, ch):
        return self.sentinel == ch

    @property
    def sentinel(self):
        return '\n'

    @property
    def start(self):
        return '_'

    def encode(self, ch):
        return self._char_to_index[ch]

    def decode(self, index):
        if index in self._index_to_char:
            return self._index_to_char[index]
        else:
            raise Exception('Unknown {}'.format(index))
            return '?'

    def __len__(self):
        return len(self._char_to_index)


class StrokeSource:
    def __init__(self, data_root):
        self._root = data_root

        self._iter = RawIterator(data_root)

    def normalize(self, x):
        return (x - x.mean(axis=0)) / np.std(x, axis=0)

    def prepare_batch(self, hand_writings, transcriptions):
        from keras.preprocessing.sequence import pad_sequences
        from keras.utils import to_categorical

        max_len = max([len(line) for line in hand_writings])
        x = pad_sequences(hand_writings, maxlen=max_len, padding='post', value=0)

        #x_norm = self.normalize(x).reshape(-1, x.shape[1], 1)
        x_norm = x.reshape(-1, x.shape[1], 1)

        seqs = []
        for tok in transcriptions:
            s = char_table.start + tok + char_table.sentinel
            encoded = [char_table.encode(ch) for ch in s]
            seqs.append(encoded)

        padded_seqs = pad_sequences(seqs, dtype=object, padding='post',
                                    value=char_table.encode(char_table.sentinel))

        y = to_categorical(padded_seqs, num_classes=len(char_table))
        y_in = y[:, :-1, :]
        y_out = y[:, 1:, :]

        return [x_norm, y_in], y_out

    def get_examples(self, batch_size=64, max_examples=1000):
        while True:
            count = 0
            hand_writings = []
            transcriptions = []
            for handwriting, transcription in self._iter.get_lines():
                hand_writings.append(handwriting)
                transcriptions.append(transcription)

                if len(hand_writings) >= batch_size:
                    yield self.prepare_batch(hand_writings, transcriptions)
                    count += batch_size
                    if count >= max_examples:
                        break

                    hand_writings = []
                    transcriptions = []

        # todo: return data, do normalization, refactor this code, random iteration of examples
        # todo: (but support deterministic one too for reproduceability)
        # todo: separate generator for word level recognition
        # todo: separate generator for images instead of strokes


charset = ''.join([chr(i) for i in range(32, 128)])
char_table = CharacterTable(charset)

source = StrokeSource('datasets/iam_online_db')

from models import SequenceToSequenceTrainer

trainer = SequenceToSequenceTrainer(decimal_table=char_table)
batch_size = 64
max_examples = 1000
trainer.fit_generator(
        source.get_examples(batch_size=batch_size, max_examples=max_examples),
        steps_per_epoch=100, epochs=10
    )
