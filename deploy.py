from tensorflow.keras.layers import Bidirectional, Dense, TimeDistributed, Input
from tensorflow.keras.models import Model
import tensorflowjs as tfjs
import tensorflow as tf
from data.data_set_home import DataSetHome, create_random_source
import os
from config import CTCConfig
from data.util import estimate_screen_width, estimate_letter_height
import json


def export_data_info(ds_home, path):
    char_table = ds_home.get_encoding_table()
    normalizer = ds_home.get_normalizer()

    provider = ds_home.get_provider()
    screen_width = estimate_screen_width(provider)
    letter_height = estimate_letter_height(provider)
    with open(path, 'w') as f:
        d = {
            'char_table': char_table._text2code,
            'horizontal_resolution': screen_width,
            'pixels_per_letter': letter_height,
            'normalization': {
                'mu': normalizer.mu,
                'sd': normalizer.sd
            }
        }

        s = json.dumps(d)
        f.write(s)


if __name__ == '__main__':
    import argparse

    cwd = os.getcwd()
    compilation_path = os.path.join(cwd, 'compiled', 'ds1')
    deploy_path = os.path.join(cwd, 'weights', 'deployed', 'blstm')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_home', type=str, default=compilation_path)
    parser.add_argument('--output_path', type=str, default=deploy_path)

    args = parser.parse_args()
    ctc_config = CTCConfig()

    data_home = DataSetHome(args.data_home, create_random_source)
    char_table = data_home.get_encoding_table()

    recurrent_layer = getattr(tf.keras.layers, ctc_config.config_dict['recurrent_layer'])
    num_cells = ctc_config.config_dict['num_cells']
    model_path = ctc_config.config_dict['weights_location']
    num_features = ctc_config.config_dict['num_features']

    output_path = args.output_path
    label_space = len(char_table) + 1

    inp = Input(shape=(None, num_features))
    lstm = Bidirectional(recurrent_layer(units=num_cells,
                                         input_shape=(None, num_features),
                                         return_sequences=True,
                                         recurrent_activation='sigmoid'))
    densor = TimeDistributed(Dense(units=label_space, activation='softmax'))

    x = inp
    x = lstm(x)
    y_pred = densor(x)

    model = Model(inp, y_pred)
    model.load_weights(model_path)

    tfjs.converters.save_keras_model(model, args.output_path)

    data_info_path = os.path.join(deploy_path, 'data_info.json')
    export_data_info(data_home, data_info_path)
