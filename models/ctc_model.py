import tensorflow as tf
from keras.layers import SimpleRNN, LSTM, Bidirectional, CuDNNLSTM, Dense, TimeDistributed, Input, Lambda, GRU, Conv1D, Reshape, MaxPool1D, Softmax
from keras.models import Sequential, Model
import numpy as np
from keras import backend as K
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import Callback
import os
from data.encodings import CharacterTable
from keras.callbacks import Callback, TensorBoard


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
    def __init__(self, recurrent_layer, embedding_size, num_cells):
        import warpctc_tensorflow

        char_table = CharacterTable()
        num_labels = len(char_table) + 1

        self.num_labels = num_labels
        self.graph_input = Input(shape=(None, embedding_size))
        self.lstm = Bidirectional(recurrent_layer(units=num_cells, input_shape=(None, embedding_size), return_sequences=True))
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
            if i >= len(gen):
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

    def fig_generator(self, train_gen, val_gen, lrate, epochs, char_table):
        optimizer = tf.train.AdamOptimizer(lrate)
        train = optimizer.minimize(self.loss)

        callback = MyCallback(self.inference_model, train_gen, val_gen, char_table)

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
    def __init__(self, recurrent_layer, embedding_size, encoding_table, num_cells=100, save_path='./trained.h5'):
        inp = Input(shape=(None, embedding_size))
        rnn_params = dict(units=num_cells, input_shape=(None, embedding_size),
                          return_sequences=True)

        num_labels = len(encoding_table) + 1

        if recurrent_layer.__name__ == 'LSTM':
            rnn_params['recurrent_activation'] = 'sigmoid'

        lstm = Bidirectional(recurrent_layer(**rnn_params))
        densor = TimeDistributed(Dense(units=num_labels, activation='softmax'))

        x = inp
        x = lstm(x)
        y_pred = densor(x)

        self.graph_input = inp
        self.graph = y_pred
        self.num_labels = num_labels
        self.encoding_table = encoding_table

        self.save_path = save_path

        self.inference_model = Model(inputs=self.graph_input, output=y_pred)

        if os.path.isfile(save_path):
            self.inference_model.load_weights(save_path)

    def compile_model(self, lrate):
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

        model.compile(optimizer=Adam(lrate), loss={'ctc': lambda y_true, y_pred: y_pred}, metrics=['acc'])
        model.summary()
        return model

    def fit_generator(self, train_gen, val_gen, lrate, epochs, batch_size=1):
        model = self.compile_model(lrate)
        validation_steps = max(1, int(len(val_gen) / batch_size))
        print('validation_steps', validation_steps)

        save_path = self.save_path
        inference_model = self.inference_model

        class SaveCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                inference_model.save_weights(save_path)

        model.fit_generator(train_gen.get_examples(),
                            steps_per_epoch=int(len(train_gen) / batch_size),
                            epochs=epochs,
                            validation_data=val_gen.get_examples(),
                            validation_steps=validation_steps,
                            callbacks=[MyCallback(inference_model, train_gen,
                                                  val_gen, self.encoding_table),
                                       TensorBoard(), SaveCallback()])


def seqlen(seq, char_table):
    s = char_table.sentinel
    for ch in seq:
        s += ch
        s += char_table.sentinel

    return len(s)


def predict(inputs, inference_model, char_table):
    x = inputs[0]
    y_hat = inference_model.predict(x)[0]

    codes = []
    for pmf in y_hat:
        index = pmf.argmax()
        codes.append(index)

    decoder = CTCOutputDecoder(char_table)
    return decoder.decode(codes)


class MyCallback(Callback):
    def __init__(self, inference_model, train_gen, val_gen, char_table):
        super().__init__()
        self._inference_model = inference_model
        self._train_gen = train_gen
        self._val_gen = val_gen
        self._char_table = char_table

    def demo(self, gen):
        counter = 0
        for inputs, y in gen.get_examples():
            counter += 1
            if counter > 5:
                break

            true = ''
            labels = y[0]
            for label in labels:
                ch = self._char_table.decode(label)
                if ch == self._char_table.sentinel:
                    true += ' '
                else:
                    true += ch

            pred = predict(inputs, self._inference_model, self._char_table)

            print(true, '->', pred)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 1 == 0:
            self.demo(self._train_gen)
            print('val')
            self.demo(self._val_gen)


class CTCOutputDecoder:
    def __init__(self, mapping):
        self._mapping = mapping

    def remove_repeated(self, labels):
        prev = -1
        res = []
        for label in labels:
            if label != prev:
                res.append(label)
                prev = label
        return res

    def remove_blanks(self, labels):
        return [label for label in labels if label != len(self._mapping)]

    def decode(self, labels):
        labels = self.remove_repeated(labels)
        labels = self.remove_blanks(labels)

        characters = [self._mapping.decode(label) for label in labels]
        return ''.join(characters)


# todo: refactor ctc_model
# todo: use predictor class instead of free predict function
# todo: create general class for callback that shows outputs afte nth epoch
