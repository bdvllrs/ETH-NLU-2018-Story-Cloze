import keras
from scripts import DefaultScript
from utils import Dataloader


class Script(DefaultScript):

    slug = 'story_cloze'

    def build_model(self, sess):
        # Graph
        entailments = keras.layers.Input(shape=(4,))  # Entailment with each sentences

    def train(self):
        training_set = Dataloader(self.config)
        training_set.load_dataset('./data/train.bin')
        training_set.load_vocab('./data/default.voc', self.config.vocab_size)

        testing_set = Dataloader(self.config, testing_data=True)
        testing_set.load_dataset('data/test.bin')
        testing_set.load_vocab('./data/default.voc', self.config.vocab_size)


