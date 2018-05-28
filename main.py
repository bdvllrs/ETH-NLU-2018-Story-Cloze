import os
from utils import Config
import argparse
from scripts import run

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Model to use")
parser.add_argument("-a", "--action", help="Action to use")
parser.add_argument("--nthreads", '-t', type=int, default=2, help="Number of threads to use")
parser.add_argument("--embedding_type", '-e', help="Type of embedding to use")
args = parser.parse_args()

config = Config('./config', args=args)

config.set('embedding_path', os.path.abspath(os.path.join(os.path.curdir, './wordembeddings.word2vec')))

run(config)
