import random
import datetime
import tensorflow as tf
import keras
from models import SentenceEmbedding
from utils import train_test
import numpy as np
import sent2vec


class Preprocess:
    """
    Preprocess to apply to the dataset
    """
    def __init__(self, sent2vec_model):
        self.sent2vec_model = sent2vec_model

    def __call__(self, word_to_index, sentence):
        sentence = self.sent2vec_model.embed_sentence(' '.join(sentence))
        return sentence


class Dataloader:
    def __init__(self):
        pass

    def __call__(self, data):
        batch = np.array(data.batch)
        last_sentences = batch[:, 3, :]
        endings = batch[:, 4, :]
        label = []
        # Randomly choose a bad ending
        for batch in range(len(batch)):
            if random.random() > 0.5:
                k = random.randint(0, len(data.dataloader)-1)
                new_data = data.dataloader.get(k, raw=True)
                new_batch = np.array(new_data.batch)
                endings[batch] = new_batch[0, 4, :]
                label.append(0)
            else:
                label.append(1)
        print(last_sentences)
        return [last_sentences, endings], np.array(label)


def keras_model(config):
    # Layers
    dense_layer_1 = keras.layers.Dense(100, activation='relu')
    dense_layer_2 = keras.layers.Dense(20, activation='relu')
    dense_layer_3 = keras.layers.Dense(2, activation='relu')

    # Inputs
    last_sentence = keras.layers.Input(shape=config.sent2vec.embedding_size)
    ending = keras.layers.Input(shape=config.sent2vec.embedding_size)

    # Graph
    inputs = keras.layers.Add()([last_sentence, ending])
    output = keras.layers.Dropout(0.2)(dense_layer_1(inputs))
    output = keras.layers.Dropout(0.2)(dense_layer_2(output))
    output = keras.layers.Dropout(0.2)(dense_layer_3(output))

    # Model
    model = keras.models.Model(inputs=[last_sentence, ending], outputs=[output])
    model.compile(optimizer="adams", loss="categorical_crossentropy", metrics=['accuracy'])

    return model


def main(config, training_set, testing_set):
    assert config.sent2vec.model is not None, "Please add sent2vec_model config value."
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(config.sent2vec.model)

    preprocess_fn = Preprocess(sent2vec_model)

    training_set.set_preprocess_fn(preprocess_fn)

    output_fn = Dataloader()
    training_set.set_output_fn(output_fn)

    print(training_set.get(1, random=True))
