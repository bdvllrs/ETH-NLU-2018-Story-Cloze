import json
import os


class Config:
    def __init__(self, file, args=None):
        filepath = os.path.abspath(os.path.join(os.path.curdir, file))
        files = os.listdir(filepath)
        files.remove('default.json')
        config_filepath = os.path.abspath(os.path.join(filepath, 'default.json'))
        with open(config_filepath, 'r') as f:
            self.config = json.load(f)
        for file in files:
            config_filepath = os.path.abspath(os.path.join(filepath, file))
            with open(config_filepath, 'r') as f:
                self.config = {**self.config, **json.load(f)}
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

