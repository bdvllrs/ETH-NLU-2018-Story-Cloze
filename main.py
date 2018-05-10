import os
from utils import Config
from scripts import scheduler_main
from Dataloader import Dataloader

config = Config('config.json')

config.set('embedding_path', os.path.abspath(os.path.join(os.path.curdir, './wordembeddings.word2vec')))

training_set = Dataloader('data/train_stories.csv')

scheduler_main(training_set, config)
