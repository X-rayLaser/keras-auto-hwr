from keras.layers import SimpleRNN, LSTM, Bidirectional, CuDNNLSTM, Dense, TimeDistributed, Input, Lambda
from keras.models import Sequential, Model
from tensorflow.nn import ctc_loss
from train_on_embeddings import get_embeddings, auto_encoder, compilation_train_source, compilation_validation_source
from data.generators import BaseGenerator
from sources.preloaded import PreLoadedSource
from tensorflow.nn import ctc_loss
from keras import backend as K


class CharacterTable:
    def __init__(self):
        index_to_char = {}
        char_to_index = {}

        num_examples = 0
        for code in range(32, 128):
            ch = chr(code)
            index = code - 32
            index_to_char[index] = ch
            char_to_index[ch] = index

            num_examples += 1

        self._char_to_index = char_to_index
        self._char_to_index[self.blank] = num_examples

        self._index_to_char = index_to_char
        self._index_to_char[num_examples] = self.blank

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

    def is_blank(self, ch):
        return self.blank == ch

    @property
    def blank(self):
        return '\t'

    def encode(self, ch):
        return self._char_to_index[ch]

    def decode(self, index):
        if index in self._index_to_char:
            return self._index_to_char[index]
        else:
            #raise Exception('Unknown {}'.format(index))
            return '?'

    def __len__(self):
        return len(self._char_to_index)


def points_source(source, num_examples):
    sin = []
    sout = []
    for strokes, t in source.get_sequences():
        new_seq = []
        if len(sout) >= num_examples:
            break
        for stroke in strokes:
            new_seq.extend(stroke.points)

        sin.append(new_seq)
        sout.append(t)

    return PreLoadedSource(sin, sout)


class CtcGenerator(BaseGenerator):
    def __init__(self, max_string_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_string_len = max_string_len

    def pad_seqsin(self, seqs_in):

        max_len = max(len(seq) for seq in seqs_in)

        res = []
        for seq in seqs_in:
            a = list(seq)
            while len(a) < max_len:
                a.append((0, 0))
            res.append(a)

        return res

    def prepare_batch(self, seqs_in, seqs_out):
        seqs_in = self.pad_seqsin(seqs_in)
        import numpy as np
        m = len(seqs_in)
        n = len(seqs_in[0])
        X = np.array(seqs_in)

        X = X.reshape((m, n, self._channels))

        classes = []
        labels = []

        input_length = np.zeros([m, 1])
        label_length = np.zeros([m, 1])

        for i, seq in enumerate(seqs_out):
            tmp = []

            for ch in seq:
                tmp.append(char_table.encode(ch))

            classes.append(tmp)

            labels.append(tmp)

            input_length[i, 0] = len(seqs_in[i])
            label_length[i, 0] = len(classes[-1])

        max_len = max(len(row) for row in labels)

        for i in range(len(labels)):
            while len(labels[i]) < max_len:
                labels[i].append(char_table.encode(char_table.blank))

        labels = np.array(labels)

        Y = np.array(classes)

        return [X, labels, input_length, label_length], Y


def seqlen(seq):
    return len(seq)
    s = 'char_table.blank'
    for ch in seq:
        s += ch
        s += char_table.blank

    return len(s)


num_train_examples = 512
num_val_examples = 256

char_table = CharacterTable()

embedding_size = 2
label_space = len(char_table)

encoder = auto_encoder.get_encoder()

train_source = points_source(compilation_train_source, num_train_examples)
val_source = points_source(compilation_validation_source, num_val_examples)


absolute_max_string_len = max([seqlen(t) for _, t in train_source.get_sequences()])
absolute_max_string_len = max(absolute_max_string_len, max([seqlen(t) for _, t in val_source.get_sequences()]))


from data.preprocessing import PreProcessor
preprocessor = PreProcessor()
train_gen = CtcGenerator(absolute_max_string_len, train_source, preprocessor, channels=embedding_size)
val_gen = CtcGenerator(absolute_max_string_len, val_source, preprocessor, channels=embedding_size)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


inp = Input(shape=(None, embedding_size))
lstm = Bidirectional(CuDNNLSTM(units=100, input_shape=(None, embedding_size), return_sequences=True))
densor = Dense(units=label_space, activation='softmax')

x = inp
x = lstm(x)
y_pred = densor(x)


labels = Input(name='the_labels',
               shape=[None], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

loss_out = Lambda(
    ctc_lambda_func, output_shape=(1,),
    name='ctc')([y_pred, labels, input_length, label_length])

from keras.optimizers import Adam, RMSprop, SGD
model = Model(inputs=[inp, labels, input_length, label_length],
              outputs=loss_out)

inference_model = Model(inputs=[inp, labels, input_length, label_length], output=y_pred)

model.compile(optimizer=Adam(0.001), loss={'ctc': lambda y_true, y_pred: y_pred}, metrics=['acc'])


def remove_repeates(s):
    return ''.join(sorted(set(s), key=s.index))


def remove_blanks(s):
    return ''.join([ch for ch in s if ch != char_table.blank])


def predict(inputs):
    y_hat = inference_model.predict(inputs)[0]

    s = ''
    for pmf in y_hat:
        index = pmf.argmax()
        ch = char_table.decode(index)

        if ch == char_table.blank:
            s += ' '

        s += ch

    s = remove_repeates(s)
    s = remove_blanks(s)

    return s


from keras.callbacks import Callback


class MyCallback(Callback):
    def demo(self, gen):
        counter = 0
        for inputs, y in gen.get_examples(1):
            counter += 1
            if counter > 5:
                break

            true = ''
            labels = y[0]
            for label in labels:
                ch = char_table.decode(label)
                if ch == char_table.blank:
                    true += ' '
                else:
                    true += ch

            pred = predict(inputs)

            print(true, '->', pred)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0 and epoch != 0:
            self.demo(train_gen)
            print('val')
            self.demo(val_gen)

from keras.callbacks import TensorBoard
batch_size = 1
validation_steps = 4
model.fit_generator(train_gen.get_examples(batch_size=batch_size),
                    steps_per_epoch=int(len(train_gen) / batch_size),
                    epochs=1500,
                    validation_data=val_gen.get_examples(batch_size),
                    validation_steps=validation_steps,
                    callbacks=[MyCallback(), TensorBoard()])

