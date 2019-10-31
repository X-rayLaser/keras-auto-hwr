from tensorflow.keras.layers import Bidirectional, Dense, TimeDistributed, Input
from tensorflow.keras.models import Model
import tensorflowjs as tfjs
import tensorflow as tf
from data.data_set_home import DataSetHome, create_random_source


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--recurrent_layer', type=str, default='LSTM')
    parser.add_argument('--num_cells', type=int, default=100)
    parser.add_argument('--data_home', type=str, default='./compiled/ds1')

    parser.add_argument('--model_path', type=str, default='./weights/blstm/blstm.h5')
    parser.add_argument('--output_path', type=str, default='./weights/deployed/blstm')

    args = parser.parse_args()

    recurrent_layer = getattr(tf.keras.layers, args.recurrent_layer)

    data_home = DataSetHome(args.data_home, create_random_source)
    char_table = data_home.get_encoding_table()

    embedding_size = 4
    num_cells = args.num_cells
    model_path = args.model_path
    output_path = args.output_path

    label_space = len(char_table) + 1

    inp = Input(shape=(None, embedding_size))
    lstm = Bidirectional(recurrent_layer(units=num_cells,
                                         input_shape=(None, embedding_size),
                                         return_sequences=True,
                                         recurrent_activation='sigmoid'))
    densor = TimeDistributed(Dense(units=label_space, activation='softmax'))

    x = inp
    x = lstm(x)
    y_pred = densor(x)

    model = Model(inp, y_pred)
    model.load_weights(model_path)

    tfjs.converters.save_keras_model(model, args.output_path)
