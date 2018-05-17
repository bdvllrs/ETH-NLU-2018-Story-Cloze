import os
from utils import Config, Sentiments
import argparse
from scripts import run
from utils import Dataloader

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Model to use")
parser.add_argument("--nthreads", '-t', type=int, default=2, help="Number of threads to use")
args = parser.parse_args()

config = Config('./config', args=args)

config.set('embedding_path', os.path.abspath(os.path.join(os.path.curdir, './wordembeddings.word2vec')))

sentiments = Sentiments(config, './data/txt_sentoken')

# training_set = Dataloader(config, './data/train_stories.csv')
# training_set.save_dataset('./data/train.bin')

training_set = Dataloader(config)

training_set.load_dataset('./data/train.bin')
# training_set.set_sentiments(sentiments)
# training_set.set_special_tokens(['<pad>', '<unk>'])
training_set.load_vocab('./default.voc', config.vocab_size)

# training_set.compute_vocab()
# training_set.save_vocab('./default.voc')

# testing_set = Dataloader(config, './data/test_stories.csv', testing_data=True)
# testing_set.save_dataset('data/test.bin')
testing_set = Dataloader(config, testing_data=True)
testing_set.load_dataset('data/test.bin')
# testing_set.set_sentiments(sentiments)

# testing_set.set_special_tokens(['<pad>', '<unk>'])
testing_set.load_vocab('./default.voc', config.vocab_size)

run(config, training_set, testing_set, sentiments)
