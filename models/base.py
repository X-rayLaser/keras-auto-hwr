import os


class BaseModel:
    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def fit_generator(self, lr, train_gen, val_gen, *args, **kwargs):
        raise NotImplementedError

    def get_inference_model(self):
        raise NotImplementedError

    def get_performance_estimator(self, num_trials):
        raise NotImplementedError


class BaseEncoderDecoder(BaseModel):
    def get_encoder(self):
        raise NotImplementedError

    def get_decoder(self):
        raise NotImplementedError

    def save(self, path):
        self.get_encoder().save_weights(os.path.join(path, 'encoder.h5'))
        self.get_decoder().save_weights(os.path.join(path, 'decoder.h5'))

    def load(self, path):
        self.get_encoder().load_weights(os.path.join(path, 'encoder.h5'))
        self.get_decoder().load_weights(os.path.join(path, 'decoder.h5'))
