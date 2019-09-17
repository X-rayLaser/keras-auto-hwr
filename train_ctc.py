from keras.layers import SimpleRNN, LSTM, Bidirectional, CuDNNLSTM, Dense, TimeDistributed, Input, Lambda, GRU, Conv1D, Reshape, MaxPool1D, Softmax
from keras.models import Sequential, Model
from data.generators import BaseGenerator
from sources.preloaded import PreLoadedSource
from keras import backend as K
from data.char_table import CharacterTable
from keras.callbacks import Callback, TensorBoard
from data.preprocessing import PreProcessor
from keras.optimizers import Adam, RMSprop, SGD
import numpy as np
from sources.compiled import CompilationSource
import tensorflow as tf


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
        m = len(seqs_in)
        n = len(seqs_in[0])
        X = np.array(seqs_in)

        X = X.reshape((m, n, self._channels))

        classes = []
        labels = []

        input_length = np.zeros([m, 1], dtype=np.int32)
        label_length = np.zeros([m, 1], dtype=np.int32)

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
                labels[i].append(char_table.encode(char_table.sentinel))

        labels = np.array(labels, dtype=np.int32)

        Y = np.array(classes)

        return [X, labels, input_length, label_length], Y


def seqlen(seq):
    s = char_table.sentinel
    for ch in seq:
        s += ch
        s += char_table.sentinel

    return len(s)


class BaseCtcModel:
    def __init__(self, recurrent_layer, num_labels, embedding_size=2, num_cells=150):
        self.graph_input = Input(shape=(None, embedding_size))
        self.rnn = Bidirectional(
            recurrent_layer(units=num_cells, input_shape=(None, embedding_size),
                            return_sequences=True)
        )
        self.densor = TimeDistributed(Dense(units=num_labels, activation=None))

        self.num_labels = num_labels

    def model_inputs(self):
        labels = Input(name='the_labels',
                       shape=[None], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        return self.graph_input, labels, input_length, label_length


class WarpCtcModel:
    def __init__(self, recurrent_layer, num_labels, embedding_size):
        import warpctc_tensorflow

        self.num_labels = num_labels
        self.graph_input = Input(shape=(None, embedding_size))
        self.lstm = Bidirectional(recurrent_layer(units=50, input_shape=(None, embedding_size), return_sequences=True))
        self.densor = TimeDistributed(Dense(units=self.num_labels, activation=None))

        self.input_lengths = tf.placeholder(tf.int32, shape=[None])
        self.label_lengths = tf.placeholder(tf.int32, shape=[None])

        self.flat_labels = tf.placeholder(tf.int32, shape=[None])

        x = self.graph_input
        x = self.lstm(x)
        linear_output = self.densor(x)

        a = Softmax()(linear_output)

        activations = tf.transpose(linear_output, perm=[1, 0, 2])

        self.loss = warpctc_tensorflow.ctc(activations, self.flat_labels, self.label_lengths,
                                           self.input_lengths, blank_label=self.num_labels-1)

        self.inference_model = Model(input=self.graph_input, output=a)

    def feeds(self, gen, batch_size):
        for i, (inputs, outputs) in enumerate(gen.get_examples(batch_size)):
            if i >= len(train_gen):
                break
            x, labels, input_length, label_length = inputs

            labels = labels.reshape(label_length[0][0])

            feed_dict = {
                self.graph_input: x,
                self.flat_labels: labels,
                self.label_lengths: label_length[0],
                self.input_lengths: input_length[0]
            }
            yield feed_dict

    def evaluate(self, gen, sess):
        losses = []
        for feed_dict in self.feeds(gen, 1):
            loss_value = sess.run(self.loss, feed_dict=feed_dict)
            losses.append(loss_value)

        return np.mean(losses)

    def fig_generator(self, train_gen, val_gen, lrate, epochs):
        optimizer = tf.train.AdamOptimizer(lrate)
        train = optimizer.minimize(self.loss)

        callback = MyCallback(self.inference_model, train_gen, val_gen)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            for epoch in range(epochs):
                print('Epoch', epoch)
                print()

                for i, feed_dict in enumerate(self.feeds(train_gen, batch_size=1)):
                    sess.run(train, feed_dict=feed_dict)

                    loss_value = sess.run(self.loss, feed_dict=feed_dict)
                    print('{} / {}, loss {}'.format(i + 1, len(train_gen), loss_value))

                print('val_loss:', self.evaluate(val_gen, sess))
                callback.on_epoch_end(epoch)


class CtcModel:
    def __init__(self, recurrent_layer, num_labels, embedding_size=2, num_cells=50):
        inp = Input(shape=(None, embedding_size))
        lstm = Bidirectional(recurrent_layer(units=num_cells, input_shape=(None, embedding_size), return_sequences=True))
        densor = TimeDistributed(Dense(units=num_labels, activation='softmax'))

        x = inp
        x = lstm(x)
        y_pred = densor(x)

        self.graph_input = inp
        self.graph = y_pred
        self.num_labels = num_labels

    def computation_graph(self):
        pass

    def compile(self, lrate):
        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args

            return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

        labels = Input(name='the_labels',
                       shape=[None], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        y_pred = self.graph
        loss_out = Lambda(
            ctc_lambda_func, output_shape=(1,),
            name='ctc')([y_pred, labels, input_length, label_length])

        model = Model(inputs=[self.graph_input, labels, input_length, label_length],
                      outputs=loss_out)

        inference_model = Model(inputs=self.graph_input, output=y_pred)

        model.compile(optimizer=Adam(lrate), loss={'ctc': lambda y_true, y_pred: y_pred}, metrics=['acc'])
        model.summary()

        return model, inference_model


def remove_repeates(codes):
    prev = -1
    res = []

    for code in codes:
        if code != prev:
            res.append(code)
            prev = code

    return res


def remove_blanks(codes):
    return [code for code in codes if code != len(char_table)]


def predict(inputs, inference_model):
    x = inputs[0]
    y_hat = inference_model.predict(x)[0]

    codes = []
    for pmf in y_hat:
        index = pmf.argmax()
        codes.append(index)

    try:
        s = ''
        for code in codes:
            try:
                ch = char_table.decode(code)
            except:
                ch = '*'
            s += ch
        print(s)
    except:
        pass

    codes = remove_repeates(codes)
    codes = remove_blanks(codes)

    s = ''
    for code in codes:
        if char_table.decode(code) == char_table.sentinel:
            break

        s += char_table.decode(code)

    return s


class MyCallback(Callback):
    def __init__(self, inference_model, train_gen, val_gen):
        super().__init__()
        self._inference_model = inference_model
        self._train_gen = train_gen
        self._val_gen = val_gen

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
                if ch == char_table.sentinel:
                    true += ' '
                else:
                    true += ch

            pred = predict(inputs, self._inference_model)

            print(true, '->', pred)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 20 == 0 and epoch != 0:
            self.demo(self._train_gen)
            print('val')
            self.demo(self._val_gen)


def dummy_source():
    sin = 'HHHH    eee  lll  lll  ooo  ,,,  www   oooo  rrr   lll  ddd'
    sout = 'Hello, world'

    char_table = CharacterTable()

    codes = [char_table.encode(ch) for ch in sin]
    from keras.utils import to_categorical

    x = to_categorical(codes, num_classes=len(char_table))

    x = x.reshape(1, len(sin), -1)

    return PreLoadedSource(x, [sout])


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./compiled')
    parser.add_argument('--max_examples', type=int, default=8)
    parser.add_argument('--lrate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--warp', type=bool, default=False)
    parser.add_argument('--recurrent_layer', type=str, default='SimpleRNN')

    args = parser.parse_args()
    print('training with following options:', args)
    char_table = CharacterTable()

    batch_size = 1
    embedding_size = 2
    num_train_examples = args.max_examples
    num_val_examples = max(1, num_train_examples // 2)
    label_space = len(char_table) + 1

    compilation_train_source = CompilationSource(
        os.path.join(args.data_path, 'train.json')
    )

    compilation_validation_source = CompilationSource(
        os.path.join(args.data_path, 'validation.json')
    )
    train_source = points_source(compilation_train_source, num_train_examples)
    val_source = points_source(compilation_validation_source, num_val_examples)
    #train_source = dummy_source()
    #val_source = dummy_source()

    preprocessor = PreProcessor()
    train_gen = CtcGenerator(train_source, preprocessor, channels=embedding_size)
    val_gen = CtcGenerator(val_source, preprocessor, channels=embedding_size)

    validation_steps = num_val_examples

    from keras import layers

    RNN_LAYER = getattr(layers, args.recurrent_layer)
    print('using layer', str(RNN_LAYER))

    if args.warp:
        ctc_model = WarpCtcModel(RNN_LAYER, label_space, embedding_size)
        ctc_model.fig_generator(train_gen, val_gen, args.lrate, args.epochs)
    else:
        ctc_model = CtcModel(RNN_LAYER, label_space, embedding_size)

        model, inference_model = ctc_model.compile(lrate=args.lrate)

        model.fit_generator(train_gen.get_examples(batch_size=batch_size),
                            steps_per_epoch=int(len(train_gen) / batch_size),
                            epochs=args.epochs,
                            validation_data=val_gen.get_examples(batch_size),
                            validation_steps=validation_steps,
                            callbacks=[MyCallback(inference_model, train_gen, val_gen), TensorBoard()])

        #model.save('./weights/blstm/blstm.h5')
