import os
from utils import Config
import argparse
from scripts import run
from Dataloader import Dataloader

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Model to use")
parser.add_argument("--nthreads", '-t', type=int, default=2, help="Number of threads to use")
args = parser.parse_args()

config = Config('config.json', args)

config.set('embedding_path', os.path.abspath(os.path.join(os.path.curdir, './wordembeddings.word2vec')))

training_set = Dataloader('data/train_stories.csv')
training_set.set_special_tokens(['<pad>', '<unk>'])
training_set.load_vocab('./default.voc', config.vocab_size)
# training_set.compute_vocab()
# training_set.save_vocab('./default.voc')

testing_set = None  # TODO

run(config, training_set, testing_set)
