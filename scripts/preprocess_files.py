import pickle
import random

import numpy as np
from tqdm import tqdm
from scripts import DefaultScript
from utils import Dataloader, SNLIDataloaderPairs


def output_fn(_, batch):
    sentence_ref = []
    sentence_pos = []
    sentence_neg = []
    for b in batch:
        sentence_ref.append(b[0][0])
        sentence_pos.append(b[0][1])
        sentence_neg.append(b[1][1])
    return [sentence_ref, sentence_pos, sentence_neg]


def preprocess_fn(line):
    output = [line['sentence1'], line['sentence2']]
    return output


class Script(DefaultScript):
    slug = 'preprocess_files'

    def train(self):
        train_set = SNLIDataloaderPairs('data/snli_1.0/snli_1.0_train.jsonl')
        train_set.load_vocab('./data/snli_vocab.dat', self.config.vocab_size)
        train_set.set_preprocess_fn(preprocess_fn)
        train_set.set_output_fn(output_fn)
        # test_set = Dataloader(self.config, 'data/test_stories.csv', testing_data=True)
        # test_set.load_dataset('data/test.bin')
        # test_set.load_vocab('./data/default.voc', self.config.vocab_size)
        # test_set.set_preprocess_fn(preprocess_fn)
        # test_set.set_output_fn(output_fn_test)

        generator_training = train_set.get_batch(1, 1)
        print(next(generator_training))
        # generator_dev = test_set.get_batch(1, 1)

    def eval(self):
        pass
