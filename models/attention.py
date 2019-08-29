from keras import Input, Model
from keras.activations import softmax
from keras.layers import SimpleRNN, Bidirectional, Dense, RepeatVector,\
    Concatenate, Activation, Dot, CuDNNGRU, Conv1D, MaxPool1D, Reshape, Dropout

from models import BaseModel
from keras.optimizers import RMSprop
import numpy as np
from estimate import AttentionModelMetric
from keras.callbacks import Callback, ReduceLROnPlateau
from models.base import BaseBeamSearch


class Seq2SeqWithAttention(BaseModel):
    def __init__(self, char_table, encoding_size, Tx, Ty):
        self.encoding_size = encoding_size
        encoder_inputs = Input(shape=(Tx, 1))

        x = encoder_inputs
        #x = Dropout(0.05)(x)
        x = Conv1D(filters=12, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPool1D()(x)

        x = Conv1D(filters=24, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPool1D()(x)
        x = Conv1D(filters=1, kernel_size=1, activation='relu')(x)
        x = Reshape(target_shape=(-1, 1))(x)
        #x = Dropout(0.05)(x)

        encoder_rnn = SimpleRNN(units=encoding_size // 2, return_sequences=True, return_state=True)
        encoder_rnn = Bidirectional(encoder_rnn)
        activations, forward_state, backward_state = encoder_rnn(x)

        activations_len = Tx // 2 // 2

        def mysoftmax(x):
            return softmax(x, axis=1)

        decoder_initial_state = Input(shape=(encoding_size, ))
        initial_y = Input(shape=(1, len(char_table)))

        decoder_rnn = SimpleRNN(units=encoding_size, return_state=True)
        densor = Dense(units=len(char_table), activation=mysoftmax)

        attention_repeator = RepeatVector(activations_len)
        attention_concatenator = Concatenate(axis=-1)
        attention_hidden_densor = Dense(units=10, activation='tanh')
        attention_output_densor = Dense(units=1, activation='relu')
        attention_softmax = Activation(mysoftmax)
        attention_dotor = Dot(axes=1)

        multi_modal_concatenator = Concatenate()
        reshapor = Reshape(target_shape=(1, len(char_table)))
        prev_s = decoder_initial_state

        outputs = []

        y_pred = initial_y
        for i in range(Ty):
            s = attention_repeator(prev_s)
            v = attention_concatenator([s, activations])
            v = attention_hidden_densor(v)
            v = attention_output_densor(v)
            alphas = attention_softmax(v)
            c = attention_dotor([alphas, activations])

            x = multi_modal_concatenator([c, y_pred])

            x, prev_s = decoder_rnn(x, initial_state=prev_s)
            y_pred = densor(x)
            outputs.append(y_pred)

            y_pred = reshapor(y_pred)

        model = Model(inputs=[encoder_inputs, decoder_initial_state, initial_y],
                      outputs=outputs)
        model.summary()
        self._model = model

        encoder_activations = Input(shape=(activations_len, encoding_size))
        prev_s = decoder_initial_state
        y_pred = initial_y

        s = attention_repeator(prev_s)
        v = attention_concatenator([s, encoder_activations])
        v = attention_hidden_densor(v)
        v = attention_output_densor(v)
        alphas = attention_softmax(v)
        c = attention_dotor([alphas, encoder_activations])

        x = multi_modal_concatenator([c, y_pred])

        x, prev_s = decoder_rnn(x, initial_state=prev_s)
        y_pred = densor(x)

        self._encoder_model = Model(input=encoder_inputs, output=activations)

        self._decoder_model = Model(inputs=[encoder_activations, decoder_initial_state, initial_y],
                                    outputs=[y_pred, prev_s])

        self._Tx = Tx
        self._Ty = Ty
        self._char_table = char_table

    def fit_generator(self, lr, train_gen, val_gen, *args, **kwargs):
        estimator = self.get_performance_estimator(8)

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                      patience=20, min_lr=0.00001)

        class MyCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 50 == 0:
                    estimator.estimate(train_gen)
                    print()
                    estimator.estimate(val_gen)
        callbacks = [MyCallback()]
        self._model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy',
                            metrics=['accuracy'])
        self._model.fit_generator(callbacks=callbacks, *args, **kwargs)

    def get_inference_model(self):
        return BeamSearchPredictor(self._encoder_model, self._decoder_model, self._char_table)

    def get_performance_estimator(self, num_trials):
        return AttentionModelMetric(self.get_inference_model(), num_trials)


class AttentionSearch(BaseBeamSearch):
    def __init__(self, encoder_state, decoder, decoder_state, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._encoder_state = encoder_state
        self._decoder = decoder
        self._initial_decoder_state = decoder_state

    def get_initial_state(self):
        return self._initial_decoder_state

    def decode_next(self, prev_y, prev_state):
        y_prob, state = self._decoder.predict([self._encoder_state, prev_state, prev_y])
        next_p = y_prob[0]
        return next_p, state


class BeamSearchPredictor:
    def __init__(self, encoder, decoder, char_table):
        self._encoder = encoder
        self._decoder = decoder
        self._char_table = char_table

    @property
    def char_table(self):
        return self._char_table

    def predict(self, inputs):
        X, initial_state, initial_y = inputs
        X = X.reshape(1, X.shape[1], 1)

        activations = self._encoder.predict(X)

        beam_search = AttentionSearch(encoder_state=activations,
                                      decoder=self._decoder,
                                      decoder_state=initial_state,
                                      char_table=self._char_table)

        return beam_search.generate_sequence()


class AttentionalPredictor:
    def __init__(self, model, char_table):
        self._model = model
        self._char_table = char_table

    @property
    def char_table(self):
        return self._char_table

    def predict(self, inputs):
        X, initial_state, initial_y = inputs
        X = X.reshape(1, X.shape[1], 1)

        predictions = self._model.predict([X, initial_state, initial_y])
        predictions = np.array(predictions)[:, 0, :]

        classes = np.argmax(predictions, axis=-1)
        return ''.join([self._char_table.decode(cls) for cls in classes])
