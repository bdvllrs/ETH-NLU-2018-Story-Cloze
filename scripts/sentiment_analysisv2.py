import datetime
import numpy as np
import keras
import os

import sent2vec
from scripts import DefaultScript


class Script(DefaultScript):

    slug = 'sentiment_analysis_v2'

    def train(self):
        main(self.config)


def training_data(sent2vec):
    sentences_file = os.path.abspath(os.path.join(os.curdir, './data/stanfordSentimentTreebank/datasetSentences.txt'))
    labels_file = os.path.abspath(os.path.join(os.curdir, './data/stanfordSentimentTreebank/sentiment_labels.txt'))
    sentences = {}
    labels = {}
    with open(sentences_file, 'r') as f:
        for k, line in enumerate(f):
            if k > 0:
                sent_id, sentence = line.rstrip().split('\t')
                sentence = sent2vec.embed_sentence(sentence)
                sentences[sent_id] = sentence
    with open(labels_file, 'r') as f:
        for k, line in enumerate(f):
            if k > 0:
                sent_id, score = line.rstrip().split('|')
                if sent_id in sentences.keys():
                    labels[sent_id] = float(score)
    sentences = list(list(zip(*sorted(sentences.items(), key=lambda w: w[0])))[1])
    labels = list(list(zip(*sorted(labels.items(), key=lambda w: w[0])))[1])
    return np.array(sentences), np.array(labels)


def model(config):
    model = keras.models.Sequential([
        keras.layers.Dense(100, activation='relu', input_shape=(config.sent2vec.embedding_size,)),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model


def main(config):
    assert config.sent2vec.model is not None, "Please add sent2vec_model config value."
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(config.sent2vec.model)

    sentiment_model = model(config)

    sentences, labels = training_data(sent2vec_model)

    verbose = 0 if not config.debug else 1
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Callbacks
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/' + timestamp + '-sentimentv2/', histogram_freq=0,
                                              batch_size=config.batch_size,
                                              write_graph=False,
                                              write_grads=True)

    model_path = os.path.abspath(
        os.path.join(os.curdir, './builds/' + timestamp + '-'))

    model_path += 'sentimentv2_checkpoint_epoch-{epoch:02d}.hdf5'
    saver = keras.callbacks.ModelCheckpoint(model_path,
                                            monitor='val_acc', verbose=verbose, save_best_only=True)

    sentiment_model.fit(sentences, labels,
                        validation_split=0.2,
                        epochs=config.n_epochs,
                        verbose=verbose,
                        callbacks=[tensorboard, saver])
