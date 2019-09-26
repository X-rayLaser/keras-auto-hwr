import json


class Config:
    def __init__(self, path='./config.json'):
        with open(path, 'r') as f:
            s = f.read()

        self.config_dict = json.loads(s)


class CTCConfig(Config):
    def __init__(self):
        super().__init__('./CTC_config.json')
