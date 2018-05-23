from os.path import dirname, basename, isfile
import glob

modules = glob.glob(dirname(__file__)+"/*.py")
files = ["scripts." + basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]


def run(config):
    print('oui')
    for file in files:
        try:
            script = __import__(file, globals(), locals(), ['*'])
            if script.Script.slug == config.model:
                print('Loading', config.model)
                script = script.Script(config)
                if config.action == 'train':
                    script.train()
                elif config.action == 'test':
                    script.test()
        except:
            pass


class DefaultScript:

    slug = 'default'

    def __init__(self, config):
        self.config = config

    def train(self):
        pass

    def test(self):
        pass
