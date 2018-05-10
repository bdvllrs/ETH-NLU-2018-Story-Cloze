import json
from os import path


class Config:
    def __init__(self, file):
        filepath = path.abspath(path.join(path.curdir, file))
        with open(filepath, 'r') as f:
            self.config = json.load(f)

    def set(self, key, value):
        self.config[key] = value

    def get(self, item):
        return self.config[item]

    def __getattr__(self, item):
        return self.config[item]

