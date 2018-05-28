from os.path import dirname, basename, isfile
import glob

modules = glob.glob(dirname(__file__)+"/*.py")
files = ["scripts." + basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]


def run(config):
    executed = False
    for file in files:
        script = __import__(file, globals(), locals(), ['*'])
        if script.Script.slug == config.model:
            if config.debug:
                print('Loading', config.model)
            script = script.Script(config)
            if config.action == 'train':
                script.train()
                executed = True
            elif config.action == 'test':
                script.test()
                executed = True
            elif config.action == 'eval':
                script.eval()
                executed = True
    if not executed and config.debug:
        print('This model or action does not exist.')


class DefaultScript:

    slug = 'default'

    def __init__(self, config):
        self.config = config

    def train(self):
        pass

    def test(self):
        pass

    def eval(self):
        pass
