import json
from os import path


class Config:
    def __init__(self, file, args=None):
        filepath = path.abspath(path.join(path.curdir, file))
        with open(filepath, 'r') as f:
            self.config = json.load(f)
        if args is not None:
            for arg, value in vars(args).items():
                if arg not in self.config.keys() or value is not None:
                    self.config[arg] = value

    def set(self, key, value):
        self.config[key] = value

    def get(self, item):
        return self.config[item]

    def __getattr__(self, item):
        return self.config[item]

