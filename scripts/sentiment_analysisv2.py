import datetime
import numpy as np
import keras
from keras import backend as K
import tensorflow as tf
import tensorflow_hub as hub
import os

from scripts import DefaultScript


class Script(DefaultScript):

    slug = 'sentiment_analysis_v2'

    def train(self):
        main(self.config)


def training_data():
    sentences_file = os.path.abspath(os.path.join(os.curdir, './data/stanfordSentimentTreebank/datasetSentences.txt'))
    labels_file = os.path.abspath(os.path.join(os.curdir, './data/stanfordSentimentTreebank/sentiment_labels.txt'))
    sentences = {}
    labels = {}
    with open(sentences_file, 'r') as f:
        for k, line in enumerate(f):
            if k > 0:
                sent_id, sentence = line.rstrip().split('\t')
                # sentence = sent2vec.embed_sentence(sentence)
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


class ElmoEmbedding:
    def __init__(self, elmo_model):
        self.elmo_model = elmo_model
        self.__name__ = "elmo_embeddings"

    def __call__(self, x):
        return self.elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)[
            "elmo"]


def model(sess, config):
    if config.debug:
        print('Importing Elmo module...')
    if config.hub.is_set("cache_dir"):
        os.environ['TFHUB_CACHE_DIR'] = config.hub.cache_dir

    elmo_model = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)

    if config.debug:
        print('Imported.')
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    elmo_embeddings = ElmoEmbedding(elmo_model)

    model = keras.models.Sequential([
        keras.layers.Lambda(elmo_embeddings, input_shape=(1,), dtype="string", output_shape=(None, 1024)),
        keras.layers.LSTM(500),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='mean_squared_error', optimizer="adam", metrics=['accuracy'])
    return model


def main(config):
    # assert config.sent2vec.model is not None, "Please add sent2vec_model config value."
    # sent2vec_model = sent2vec.Sent2vecModel()
    # sent2vec_model.load_model(config.sent2vec.model)

    sess = tf.Session()
    K.set_session(sess)  # Set to keras backend

    sentiment_model = model(sess, config)

    sentences, labels = training_data()

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
