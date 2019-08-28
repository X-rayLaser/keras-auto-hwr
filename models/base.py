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
