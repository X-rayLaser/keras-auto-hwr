from keras import Input, Model
from keras.activations import softmax
from keras.layers import SimpleRNN, Bidirectional, Dense, RepeatVector, Concatenate, Activation, Dot

from models import BaseModel


class Seq2SeqWithAttention(BaseModel):
    def __init__(self, char_table, encoding_size=32, Tx=400, Ty=70):
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

        self._model = model
        self._Tx = Tx
        self._Ty = Ty
        self._char_table = char_table

    def fit_generator(self):
        # todo: implement this method and get_inference_model as well
        pass