from keras.layers import Conv1D, MaxPool1D
from config import Config
import sys


class ConvolutionLayer:
    def __init__(self, filters, kernel_size):
        self.filters = filters
        self.kernel_size = kernel_size

    def back_track(self, sequential_id):
        return list(range(sequential_id, sequential_id + self.kernel_size))

    def output_size(self, input_size):
        return input_size - self.kernel_size + 1

    def create(self):
        return Conv1D(filters=self.filters, kernel_size=self.kernel_size,
                      activation='relu')


class PoolingLayer:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def back_track(self, sequential_id):
        start_index = sequential_id * self.kernel_size
        return list(range(start_index, start_index + self.kernel_size))

    def output_size(self, input_size):
        return input_size // self.kernel_size

    def create(self):
        return MaxPool1D(self.kernel_size)


class EncoderSpec:
    def __init__(self, input_size):
        self._input_size = input_size
        self._layers = []

    @staticmethod
    def from_config(input_size):
        this_module = sys.modules[__name__]
        conf = Config()

        spec = EncoderSpec(input_size)

        layers = conf.config_dict["attention_model"]["encoder_spec"]
        for layer in layers:
            name = layer["name"]
            params = layer["params"]
            cls = getattr(this_module, name)
            spec._layers.append(cls(**params))

        return spec

    def get_graph(self, initial_x):
        tensor = initial_x
        for layer in self._layers:
            tensor = layer.create()(tensor)

        return tensor

    def add_conv_layer(self, filters, kernel_size):
        self._layers.append(ConvolutionLayer(filters, kernel_size))

    def add_pooling_layer(self, pool_size=2):
        self._layers.append(PoolingLayer(pool_size))

    def output_size(self):
        size = self._input_size

        for layer in self._layers:
            size = layer.output_size(size)

        return size

    def back_track(self, sequential_id):
        mapped_ids = [sequential_id]

        for layer in reversed(self._layers):
            back_ids = []
            for elem_id in mapped_ids:
                back_ids.extend(layer.back_track(elem_id))
            mapped_ids = list(set(back_ids))

        return mapped_ids