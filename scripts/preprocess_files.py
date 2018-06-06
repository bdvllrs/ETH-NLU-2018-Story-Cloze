import pickle
import random

import numpy as np
from tqdm import tqdm
from scripts import DefaultScript
from utils import Dataloader


def output_fn(data):
    batch = np.array(data.batch)
    return [list(batch[0, 0]), list(batch[0, 1]), list(batch[0, 2]), list(batch[0, 3]), list(batch[0, 4])]


def output_fn_test(data):
    batch = np.array(data.batch)
    return [list(batch[0, 0]), list(batch[0, 1]), list(batch[0, 2]), list(batch[0, 3]), list(batch[0, 4]),
            list(batch[0, 5]), int(data.label[0])-1]


class Preprocess:
    """
    Preprocess to apply to the dataset
    """

    def __init__(self, sent2vec_model):
        self.sent2vec_model = sent2vec_model

    def __call__(self, word_to_index, sentence):
        sentence = self.sent2vec_model.embed_sentence(' '.join(sentence))
        return sentence


class Script(DefaultScript):
    slug = 'preprocess_files'

    def train(self):
        import sent2vec
        assert self.config.sent2vec.model is not None, "Please add sent2vec_model config value."
        sent2vec_model = sent2vec.Sent2vecModel()
        sent2vec_model.load_model(self.config.sent2vec.model)

        preprocess_fn = Preprocess(sent2vec_model)
        train_set = Dataloader(self.config, 'data/train_stories.csv')
        train_set.load_dataset('data/train.bin')
        train_set.set_preprocess_fn(preprocess_fn)
        train_set.load_vocab('./data/default.voc', self.config.vocab_size)
        train_set.set_output_fn(output_fn)
        # test_set = Dataloader(self.config, 'data/test_stories.csv', testing_data=True)
        # test_set.load_dataset('data/test.bin')
        # test_set.load_vocab('./data/default.voc', self.config.vocab_size)
        # test_set.set_preprocess_fn(preprocess_fn)
        # test_set.set_output_fn(output_fn_test)

        generator_training = train_set.get_batch(1, 1)
        # generator_dev = test_set.get_batch(1, 1)
        with open(
                "/run/media/bdvllrs/Data/Documents/ETH/Natural Language Understanding/Project 2/Story Cloze/data/train_pp.pickle",
                "wb") as file:
            for b in tqdm(generator_training, total=len(train_set)):
                pickle.dump(b, file)

    def eval(self):
        pass
