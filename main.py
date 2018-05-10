import os
from utils import Config
import argparse
from scripts import run
from Dataloader import Dataloader

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Model to use")
args = parser.parse_args()

config = Config('config.json', args)

config.set('embedding_path', os.path.abspath(os.path.join(os.path.curdir, './wordembeddings.word2vec')))

training_set = Dataloader('data/train_stories.csv')

run(config, training_set, None)
