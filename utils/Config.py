import json
import os


class Config:
    def __init__(self, file=None, config=None, args=None):
        if file is not None:
            filepath = os.path.abspath(os.path.join(os.path.curdir, file))
            files = os.listdir(filepath)
            files.remove('default.json')
            config_filepath = os.path.abspath(os.path.join(filepath, 'default.json'))
            with open(config_filepath, 'r') as f:
                self.config = json.load(f)
            for file in files:
                if file[-4:] == 'json':
                    config_filepath = os.path.abspath(os.path.join(filepath, file))
                    with open(config_filepath, 'r') as f:
                        self.config = {**self.config, **json.load(f)}
        elif config is not None:
            self.config = config
        if args is not None:
            for arg, value in vars(args).items():
                if arg not in self.config.keys() or value is not None:
                    self.config[arg] = value

    def set(self, key, value):
        self.config[key] = value

    def get(self, item):
        if type(self.config[item]) == dict:
            return Config(config=self.config[item])
        return self.config[item]

    def __str__(self):
        return str(self.config)

    def __getattr__(self, item):
        return self.get(item)
