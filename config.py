import json


class Config:
    def __init__(self):
        with open('./config.json', 'r') as f:
            s = f.read()

        self.config_dict = json.loads(s)
