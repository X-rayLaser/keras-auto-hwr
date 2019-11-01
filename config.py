import json
import os

cwd = os.path.abspath(os.getcwd())
ctc_config_path = os.path.join(cwd, 'CTC_config.json')


class Config:
    def __init__(self, path=None):
        if path is None:
            path = os.path.join(cwd, 'config.json')
        with open(path, 'r') as f:
            s = f.read()

        self.config_dict = json.loads(s)


class CTCConfig(Config):
    def __init__(self):
        super().__init__(ctc_config_path)
