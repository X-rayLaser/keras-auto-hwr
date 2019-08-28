from keras import Input, Model
from keras.activations import softmax
from keras.layers import SimpleRNN, Bidirectional, Dense, RepeatVector, Concatenate, Activation, Dot, CuDNNGRU

from models import BaseModel
from keras.optimizers import RMSprop
import numpy as np
from estimate import AttentionModelMetric
from keras.callbacks import Callback


class Seq2SeqWithAttention(BaseModel):
    def __init__(self, char_table, encoding_size, Tx=400, Ty=70):
        self.encoding_size = encoding_size
        encoder_inputs = Input(shape=(Tx, 1))

        encoder_rnn = SimpleRNN(units=encoding_size // 2, return_sequences=True, return_state=True)
        encoder_rnn = Bidirectional(encoder_rnn)

        x = encoder_inputs
        activations, forward_state, backward_state = encoder_rnn(x)

        def mysoftmax(x):
            return softmax(x, axis=1)

        decoder_initial_state = Input(shape=(encoding_size, ))

        decoder_rnn = SimpleRNN(units=encoding_size, return_state=True)
        densor = Dense(units=len(char_table), activation=mysoftmax)

        attention_repeator = RepeatVector(Tx)
        attention_concatenator = Concatenate(axis=-1)
        attention_hidden_densor = Dense(units=10, activation='tanh')
        attention_output_densor = Dense(units=1, activation='relu')
        attention_softmax = Activation(mysoftmax)
        attention_dotor = Dot(axes=1)

        prev_s = decoder_initial_state

        outputs = []

        for i in range(Ty):
            s = attention_repeator(prev_s)
            v = attention_concatenator([s, activations])
            v = attention_hidden_densor(v)
            v = attention_output_densor(v)
            alphas = attention_softmax(v)
            c = attention_dotor([alphas, activations])

            x, prev_s = decoder_rnn(c, initial_state=prev_s)
            y_pred = densor(x)
            outputs.append(y_pred)

        model = Model(inputs=[encoder_inputs, decoder_initial_state], outputs=outputs)
        model.summary()
        self._model = model
        self._Tx = Tx
        self._Ty = Ty
        self._char_table = char_table

    def fit_generator(self, lr, train_gen, val_gen, *args, **kwargs):
        estimator = self.get_performance_estimator(8)

        class MyCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 5 == 0:
                    estimator.estimate(train_gen)
                    print()
                    estimator.estimate(val_gen)
        callbacks = [MyCallback()]
        self._model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy',
                            metrics=['accuracy'])
        self._model.fit_generator(callbacks=callbacks, *args, **kwargs)

    def get_inference_model(self):
        return AttentionalPredictor(self._model, self.encoding_size, self._char_table)

    def get_performance_estimator(self, num_trials):
        return AttentionModelMetric(self.get_inference_model(), num_trials)


class AttentionalPredictor:
    def __init__(self, model, encoding_size, char_table):
        self._model = model
        self._char_table = char_table
        self._encoding_size = encoding_size

    @property
    def char_table(self):
        return self._char_table

    def predict(self, hand_writing):
        hand_writing = hand_writing.reshape(1, hand_writing.shape[1], 1)
        state = np.zeros((1, self._encoding_size))

        predictions = self._model.predict([hand_writing, state])
        predictions = np.array(predictions)[:, 0, :]

        classes = np.argmax(predictions, axis=-1)
        return ''.join([self._char_table.decode(cls) for cls in classes])
