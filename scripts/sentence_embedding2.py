import random
import datetime
import tensorflow as tf
import keras
from models import SentenceEmbedding
from utils import train_test
import numpy as np
import sent2vec


class Dataloader:
    def __init__(self):
        pass

    def __call__(self, data):
        print(data.batch)
        return data


def main(config, training_set, testing_set):
    assert config.sent2vec.model is not None, "Please add sent2vec_model config value."
    model = sent2vec.Sent2vecModel()
    model.load_model(config.sent2vec.model)

    output_fn = Dataloader()
    training_set.set_output_fn(output_fn)

    print(training_set.get(1, random=True))
