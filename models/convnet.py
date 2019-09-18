from models.base import BaseEncoderDecoder
from keras.layers import Input, Conv1D, MaxPool1D, Dense, Flatten, Dropout
from keras.regularizers import l2
from keras.models import Model
import numpy as np


class ConvolutionalRecognizer(BaseEncoderDecoder):
    def __init__(self, vocab, input_len, channels, embedding_size=256):
        inp = Input(shape=(input_len, channels))
        x = inp

        x = Conv1D(filters=channels, kernel_size=(3,), padding='same', activation='relu')(x)
        x = Conv1D(filters=channels * 2, kernel_size=(3,), padding='same', activation='relu')(x)
        x = MaxPool1D()(x)
        x = Conv1D(filters=channels * 4, kernel_size=(3,), padding='same', activation='relu')(x)
        x = MaxPool1D()(x)

        x = Flatten()(x)
        x = Dense(units=256, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.25)(x)
        embedding = Dense(units=embedding_size, activation='relu', kernel_regularizer=l2(0.01))(x)

        self._encoder = Model(input=inp, output=embedding)

        decoder_inp = Input(shape=(embedding_size,))

        x = decoder_inp
        x = Dropout(0.125)(x)
        y_pred = Dense(units=len(vocab), activation='softmax', kernel_regularizer=l2(0.01))(x)

        self._decoder = Model(input=decoder_inp, output=y_pred)

        x = inp
        x = self._encoder(x)
        y_pred = self._decoder(x)
        self._model = Model(input=inp, output=y_pred)
        self._vocab = vocab

    def get_encoder(self):
        return self._encoder

    def get_decoder(self):
        return self._decoder

    def fit_generator(self, lr, train_gen, val_gen, *args, **kwargs):
        from keras.callbacks import Callback
        from keras.optimizers import Adam

        predictor = self.get_inference_model()

        vocab = self._vocab

        class MyCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 10 == 0:
                    counter = 0
                    for x, transcription_1hot in train_gen.get_examples(1):
                        if counter > 10:
                            break

                        transcription = vocab.decode(np.argmax(transcription_1hot))
                        s = predictor.predict(x)
                        print(transcription, '->', s)
                        counter += 1

                    # and now on test data
                    print()
                    counter = 0
                    for x, transcription_1hot in val_gen.get_examples(1):
                        if counter > 10:
                            break

                        transcription = vocab.decode(np.argmax(transcription_1hot))
                        s = predictor.predict(x)
                        print(transcription, '->', s)
                        counter += 1

        callbacks = [MyCallback()]
        self._model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy',
                            metrics=['categorical_crossentropy'])
        self._model.fit_generator(callbacks=callbacks, *args, **kwargs)

    def get_inference_model(self):
        model = self._model
        vocab = self._vocab

        class Predictor:
            def predict(self, x):
                y_pred = model.predict(x)[0]
                cls = np.argmax(y_pred)
                s = vocab.decode(cls)
                return s

        return Predictor()


from data.generators import BaseGenerator
from keras.utils import to_categorical


class ConvNetGenerator(BaseGenerator):
    def __init__(self, vocab, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vocab = vocab

    def prepare_batch(self, seqs_in, seqs_out):
        m = len(seqs_in)
        n = len(seqs_in[0])
        X = np.array(seqs_in)
        X = X.reshape((m, n, self._channels))

        classes = []
        for s in seqs_out:
            classes.append(self._vocab.encode(s))

        Y = to_categorical(classes, num_classes=len(self._vocab))

        return X, Y
