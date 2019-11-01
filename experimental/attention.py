from keras import Input, Model
from keras.activations import softmax
from keras.layers import SimpleRNN, Bidirectional, Dense, RepeatVector,\
    Concatenate, Activation, Dot, Reshape

from models import BaseModel
from keras.optimizers import RMSprop
import numpy as np
from experimental.estimate import AttentionModelMetric
from keras.callbacks import Callback, ReduceLROnPlateau
from algorithms.beam_search import BaseBeamSearch
from experimental.encoder_spec import EncoderSpec
from keras.regularizers import l2


class Seq2SeqWithAttention(BaseModel):
    def __init__(self, char_table, encoding_size, Tx, Ty, channels):

        def mysoftmax(x):
            return softmax(x, axis=1)

        self.encoding_size = encoding_size
        self._Tx = Tx
        self._Ty = Ty
        self._char_table = char_table
        self._mysoftmax = mysoftmax

        self._encoder_spec = EncoderSpec.from_config(self._Tx)

        encoder, activations_len = self.encoder_model(Tx, channels, encoding_size)
        attention = self.attention_model(activations_len, mysoftmax)
        decoder = self.one_step_decoder(attention, activations_len)

        encoder_inputs = Input(shape=(Tx, channels))
        decoder_initial_state = Input(shape=(self.encoding_size,))
        initial_y = Input(shape=(1, len(self._char_table)))
        reshapor = Reshape(target_shape=(1, len(self._char_table)))

        x = encoder_inputs

        activations = encoder(x)

        outputs = []
        prev_state = decoder_initial_state
        prev_y = initial_y
        for i in range(self._Ty):
            prev_y, prev_state, alphas = decoder([activations, prev_state, prev_y])
            outputs.append(prev_y)
            prev_y = reshapor(prev_y)

        model = Model(inputs=[encoder_inputs, decoder_initial_state, initial_y],
                      outputs=outputs)

        self._model = model

        self._encoder_model = encoder
        self._decoder_model = decoder

    def one_step_decoder(self, attention, activations_len):
        activations = Input(shape=(activations_len, self.encoding_size))

        decoder_initial_state = Input(shape=(self.encoding_size,))
        initial_y = Input(shape=(1, len(self._char_table)))
        decoder_rnn = SimpleRNN(units=self.encoding_size, return_state=True, kernel_regularizer=l2(0.0001))
        densor = Dense(units=len(self._char_table), activation=self._mysoftmax, kernel_regularizer=l2(0.0001))
        attention_dotor = Dot(axes=1)
        multi_modal_concatenator = Concatenate()

        prev_s = decoder_initial_state

        y_pred = initial_y
        alphas = attention([prev_s, activations])
        c = attention_dotor([alphas, activations])

        x = multi_modal_concatenator([c, y_pred])

        x, prev_s = decoder_rnn(x, initial_state=prev_s)
        y_pred = densor(x)

        return Model(inputs=[activations, decoder_initial_state, initial_y],
                     outputs=[y_pred, prev_s, alphas])

    def attention_model(self, activations_len, mysoftmax):
        previous_state = Input(shape=(self.encoding_size,))
        activations = Input(shape=(activations_len, self.encoding_size))

        attention_repeator = RepeatVector(activations_len)
        attention_concatenator = Concatenate(axis=-1)
        attention_hidden_densor = Dense(units=10, activation='tanh')
        attention_output_densor = Dense(units=1, activation='relu')
        attention_softmax = Activation(mysoftmax)

        s = attention_repeator(previous_state)
        v = attention_concatenator([s, activations])
        v = attention_hidden_densor(v)
        v = attention_output_densor(v)
        alphas = attention_softmax(v)
        return Model(inputs=[previous_state, activations], output=alphas)

    def encoder_model(self, Tx, channels, encoding_size):
        encoder_inputs = Input(shape=(Tx, channels))
        x = encoder_inputs
        #x = self._encoder_spec.get_graph(x)
        #x = Reshape(target_shape=(-1, 1))(x)
        encoder_rnn = SimpleRNN(units=encoding_size // 2, return_sequences=True, return_state=True,
                                kernel_regularizer=l2(0.001))
        encoder_rnn = Bidirectional(encoder_rnn)
        activations, forward_state, backward_state = encoder_rnn(x)
        #activations_len = self._encoder_spec.output_size()
        activations_len = self._Tx

        return Model(input=encoder_inputs, output=activations), activations_len

    def fit_generator(self, lr, train_gen, val_gen, *args, **kwargs):
        estimator = self.get_performance_estimator(8)

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                      patience=20, min_lr=0.00001)

        debug_predictor = DebugPredictor(self._encoder_model,
                                         self._decoder_model,
                                         self._char_table)

        mapper = self._encoder_spec

        debug = AttentionDebugCallback(train_gen, debug_predictor, mapper)

        class MyCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 10 == 0:
                    estimator.estimate(train_gen)
                    print()
                    estimator.estimate(val_gen)

        callbacks = [MyCallback()]
        self._model.compile(optimizer=RMSprop(lr=lr, clipnorm=10, clipvalue=10), loss='categorical_crossentropy',
                            metrics=['categorical_crossentropy'])
        self._model.fit_generator(callbacks=callbacks, *args, **kwargs)

    def get_inference_model(self):
        return BeamSearchPredictor(self._encoder_model, self._decoder_model, self._char_table)

    def get_performance_estimator(self, num_trials):
        return AttentionModelMetric(self.get_inference_model(), num_trials)

    def get_spec(self):
        pass


class AttentionSearch(BaseBeamSearch):
    def __init__(self, encoder_state, decoder, decoder_state, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._encoder_state = encoder_state
        self._decoder = decoder
        self._initial_decoder_state = decoder_state

    def get_initial_state(self):
        return self._initial_decoder_state

    def decode_next(self, prev_y, prev_state):
        y_prob, state, attention_coefficients = self._decoder.predict(
            [self._encoder_state, prev_state, prev_y]
        )
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

        activations = self._encoder.predict(X)

        beam_search = AttentionSearch(encoder_state=activations,
                                      decoder=self._decoder,
                                      decoder_state=initial_state,
                                      char_table=self._char_table)

        return beam_search.generate_sequence()


class DebugPredictor(BeamSearchPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._coefficients = []

    def predict(self, inputs):
        self._coefficients = []

        X, initial_state, initial_y = inputs
        X = X.reshape(1, X.shape[1], 1)

        activations = self._encoder.predict(X)

        prev_state = initial_state
        prev_y = initial_y
        s = ''
        ch = self.char_table.start
        max_len = 150
        while ch != self.char_table.sentinel and len(s) < max_len:
            prev_y, prev_state, alphas = self._decoder.predict([activations, prev_state, prev_y])

            self._coefficients.append((s, alphas.squeeze().tolist()))

            prev_y = prev_y.reshape((-1, 1, len(self.char_table)))

            cls = np.argmax(prev_y[0][-1])
            ch = self.char_table.decode(cls)
            s += ch

        return s

    def history(self):
        return self._coefficients


class Debugger:
    def __init__(self):
        pass


class AttentionDebugCallback(Callback):
    def __init__(self, gen, predictor, mapper):
        super().__init__()
        self._gen = gen
        self._predictor = predictor
        self._mapper = mapper

    def visualize_attention(self, strokes, alphas):
        from PIL import Image, ImageDraw

        points = []
        for stroke in strokes:
            for p in stroke.points:
                points.append(p)

        width = max([x for x, y in points])
        height = max([y for x, y in points])

        a = np.zeros((height, width), dtype=np.uint8)

        im = Image.fromarray(a, mode='L')

        canvas = ImageDraw.ImageDraw(im, mode='L')

        for i in range(len(alphas)):
            indices = self._mapper.back_track(i)
            indices = sorted(indices)
            prev_point = None

            for index in indices:
                x, y = points[index]
                intensity = alphas[i]

                if prev_point:
                    canvas.line((prev_point, (x, y)), width=12, fill=int(round(255 * intensity)))
                prev_point = (x, y)
        im.show()

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 1 and epoch % 150 == 0:
            i = 0
            for original, processed in self._gen.debug_examples():
                i += 1
                if i > 2:
                    break

                [x_norm, initial_state, initial_y], final_y = processed
                s = self._predictor.predict([x_norm, initial_state, initial_y])
                print('full output', s)
                for prefix, coefficients in self._predictor.history():
                    print(prefix)
                    self.visualize_attention(original[0], coefficients)
                    input('Press any key')
