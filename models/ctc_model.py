from keras.layers import Bidirectional, Dense, TimeDistributed, Input, Lambda
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
import os
from keras.callbacks import Callback, TensorBoard
from tests.test_predictor import DebugPredictor
from data.example_adapters import CTCAdapter
from algorithms.token_passing import TokenPassing


class CtcModel:
    def __init__(self, recurrent_layer, embedding_size, encoding_table, num_cells=100, save_path=None):
        if save_path is None:
            save_path = os.path.join(os.getcwd(), 'trained.h5')
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

        save_path = self.save_path
        inference_model = self.inference_model

        class SaveCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                inference_model.save_weights(save_path)

        class_predictor = BestPathClassPredictor(inference_model)

        model.fit_generator(train_gen.get_examples(),
                            steps_per_epoch=int(len(train_gen) / batch_size),
                            epochs=epochs,
                            validation_data=val_gen.get_examples(),
                            validation_steps=validation_steps,
                            callbacks=[MyCallback(class_predictor, train_gen,
                                                  val_gen, self.encoding_table),
                                       TensorBoard(), SaveCallback()])


class BestPathClassPredictor:
    def __init__(self, model):
        self._model = model

    def predict(self, x):
        y_hat = self._model.predict(x)[0]

        codes = []
        for pmf in y_hat:
            index = pmf.argmax()
            codes.append(index)

        return codes


class TokenPassingPredictor:
    def __init__(self, model, word_dict, encoding_table):
        self._model = model
        self._word_dict = word_dict
        self._encoding_table = encoding_table

    def predict(self, x):
        pmfs = self._model.predict(x)[0]

        from algorithms.token_passing import token_passing_cpp

        token_passing_cpp(pmfs, self._encoding_table)

        dictionary = self._word_dict
        algo = TokenPassing(dictionary, pmfs, self._encoding_table)

        return algo.decode()


class MyCallback(Callback):
    def __init__(self, class_predictor, train_gen, val_gen, char_table):
        super().__init__()
        self._class_predictor = class_predictor
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

            adapter = CTCAdapter(self._char_table.sentinel)
            decoder = CTCOutputDecoder(self._char_table)

            predictor = DebugPredictor(self._class_predictor, adapter, decoder)
            pred = predictor.predict(inputs[0][0])

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
