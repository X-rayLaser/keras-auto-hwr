from keras.layers import SimpleRNN, LSTM, Bidirectional, CuDNNLSTM, Dense, TimeDistributed, Input, Lambda, GRU, Conv1D, Reshape, MaxPool1D
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
import warpctc_tensorflow


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


class CTCTrainer:
    def __init__(self, model):
        self._model = model

    def fit_generator(self, train_gen, val_gen):
        model = self._model
        model.fit_generator(train_gen.get_examples(batch_size=batch_size),
                            steps_per_epoch=int(len(train_gen) / batch_size),
                            epochs=args.epochs,
                            validation_data=val_gen.get_examples(batch_size),
                            validation_steps=validation_steps,
                            callbacks=[MyCallback(), TensorBoard()])


class WarpTrainer(CTCTrainer):
    def fit_generator(self, train_gen, val_gen):
        pass



class CtcModel:
    def __init__(self, num_labels, embedding_size=2, num_cells=50):
        inp = Input(shape=(None, embedding_size))
        lstm = Bidirectional(GRU(units=num_cells, input_shape=(None, embedding_size), return_sequences=True))
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

        inference_model = Model(inputs=[self.graph_input, labels, input_length, label_length], output=y_pred)

        model.compile(optimizer=Adam(lrate), loss={'ctc': lambda y_true, y_pred: y_pred}, metrics=['acc'])
        model.summary()

        return model, inference_model

    def train_warp(self, train_gen, lrate, epochs):
        inp = Input(shape=(None, embedding_size))
        lstm = Bidirectional(GRU(units=50, input_shape=(None, embedding_size), return_sequences=True))
        densor = TimeDistributed(Dense(units=self.num_labels, activation=None))

        x = inp
        x = lstm(x)
        y_pred = densor(x)
        self.graph_input = inp

        input_lengths = tf.placeholder(tf.int32, shape=[None])
        label_lengths = tf.placeholder(tf.int32, shape=[None])

        flat_labels = tf.placeholder(tf.int32, shape=[None])

        activations = tf.transpose(y_pred, perm=[1, 0, 2])

        loss = warpctc_tensorflow.ctc(activations, flat_labels, label_lengths,
                                      input_lengths, blank_label=self.num_labels-1)
        optimizer = tf.train.AdamOptimizer(lrate)
        train = optimizer.minimize(loss)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            for epoch in range(epochs):
                print('Epoch', epoch)
                print()

                for i, (inputs, outputs) in enumerate(train_gen.get_examples(1)):
                    if i >= len(train_gen):
                        break
                    x, labels, input_length, label_length = inputs

                    labels = labels.reshape(label_length[0][0])

                    feed_dict = {
                        self.graph_input: x,
                        flat_labels: labels,
                        label_lengths: label_length[0],
                        input_lengths: input_length[0]
                    }
                    sess.run(train, feed_dict=feed_dict)

                    loss_value = sess.run(loss, feed_dict=feed_dict)
                    print('{} / {}, loss {}'.format(i + 1, len(train_gen), loss_value))


def remove_repeates(codes):
    return sorted(set(codes), key=codes.index)


def remove_blanks(codes):
    return [code for code in codes if code != len(char_table)]


def predict(inputs):
    y_hat = inference_model.predict(inputs)[0]

    codes = []
    for pmf in y_hat:
        index = pmf.argmax()
        codes.append(index)

    codes = remove_repeates(codes)
    codes = remove_blanks(codes)

    s = ''
    for code in codes:
        if char_table.decode(code) == char_table.sentinel:
            break

        s += char_table.decode(code)

    return s


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
                if ch == char_table.sentinel:
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


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./compiled')
    parser.add_argument('--max_examples', type=int, default=8)
    parser.add_argument('--lrate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()
    char_table = CharacterTable()

    batch_size = 1
    embedding_size = 2
    num_train_examples = args.max_examples
    num_val_examples = min(8, num_train_examples // 8)
    label_space = len(char_table) + 1

    compilation_train_source = CompilationSource(
        os.path.join(args.data_path, 'train.json')
    )

    compilation_validation_source = CompilationSource(
        os.path.join(args.data_path, 'validation.json')
    )
    train_source = points_source(compilation_train_source, num_train_examples)
    val_source = points_source(compilation_validation_source, num_val_examples)

    preprocessor = PreProcessor()
    train_gen = CtcGenerator(train_source, preprocessor, channels=embedding_size)
    val_gen = CtcGenerator(val_source, preprocessor, channels=embedding_size)

    validation_steps = num_val_examples

    ctc_model = CtcModel(label_space, embedding_size)

    """
    model, inference_model = ctc_model.compile(lrate=args.lrate)

    model.fit_generator(train_gen.get_examples(batch_size=batch_size),
                        steps_per_epoch=int(len(train_gen) / batch_size),
                        epochs=args.epochs,
                        validation_data=val_gen.get_examples(batch_size),
                        validation_steps=validation_steps,
                        callbacks=[MyCallback(), TensorBoard()])

    model.save('./weights/blstm/blstm.h5')
    """
    ctc_model.train_warp(train_gen, args.lrate, args.epochs)
