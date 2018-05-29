import datetime
import random

import numpy as np
import keras
from keras import backend as K
import os
import tensorflow_hub as hub
import tensorflow as tf
from scripts import DefaultScript
from utils import Dataloader


class ElmoEmbedding:
    def __init__(self, elmo_model, embedding_type):
        self.elmo_model = elmo_model
        self.embedding_type = embedding_type
        self.__name__ = "elmo_embeddings"

    def __call__(self, x):
        if self.embedding_type['name'] == "elmo":
            return self.elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)[
                "default"]
        if self.embedding_type['name'] == "use":
            return self.elmo_model(tf.squeeze(tf.cast(x, tf.string)))


def output_fn(data):
    batch = np.array(data.batch)
    sentences1 = []
    sentences2 = []
    sentences3 = []
    sentences4 = []
    sentences5 = []
    label = []
    for b in batch:
        sentences1.append(" ".join(b[0]))
        sentences2.append(" ".join(b[1]))
        sentences3.append(" ".join(b[2]))
        sentences4.append(" ".join(b[3]))
        if random.random() > 0.5:
            sentences5.append(" ".join(b[4]))
            label.append(1)
        else:
            k = random.randint(0, len(data.dataloader))
            random_story = data.dataloader.get(k, 1, raw=True).batch[0]
            sentences5.append(" ".join(random_story[4]))
            if random_story[4] == b[4]:
                label.append(1)
            else:
                label.append(0)
    return [np.array(sentences1), np.array(sentences2), np.array(sentences3), np.array(sentences4),
            np.array(sentences5)], np.array(label)


def output_fn_test(data):
    batch = np.array(data.batch)
    sentences1 = []
    sentences2 = []
    sentences3 = []
    sentences4 = []
    sentences5 = []
    label = []
    for b in batch:
        sentences1.append(" ".join(b[0]))
        sentences2.append(" ".join(b[1]))
        sentences3.append(" ".join(b[2]))
        sentences4.append(" ".join(b[3]))
        if random.random() > 0.5:
            sentences5.append(" ".join(b[4]))
            label.append(2 - int(b[6][0]))
        else:
            sentences5.append(" ".join(b[5]))
            label.append(int(b[6][0]) - 1)
    return [np.array(sentences1), np.array(sentences2), np.array(sentences3), np.array(sentences4),
            np.array(sentences5)], np.array(label)


class Script(DefaultScript):
    slug = 'reorder_full_rd_elmo'

    def train(self):
        train_set = Dataloader(self.config, 'data/train_stories.csv')
        test_set = Dataloader(self.config, 'data/test_stories.csv', testing_data=True)
        train_set.load_dataset('data/train.bin')
        train_set.load_vocab('./data/default.voc', self.config.vocab_size)
        test_set.load_dataset('data/test.bin')
        test_set.load_vocab('./data/default.voc', self.config.vocab_size)
        test_set.set_output_fn(output_fn_test)
        train_set.set_output_fn(output_fn)

        generator_training = train_set.get_batch(self.config.batch_size, self.config.n_epochs)
        generator_dev = test_set.get_batch(self.config.batch_size, self.config.n_epochs)

        # Initialize tensorflow session
        sess = tf.Session()
        K.set_session(sess)  # Set to keras backend

        keras_model = self.build_graph(sess)

        verbose = 0 if not self.config.debug else 1
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Callbacks
        tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/' + timestamp + '-reorder-full-rd-elmo/',
                                                  histogram_freq=0,
                                                  batch_size=self.config.batch_size,
                                                  write_graph=False,
                                                  write_grads=True)

        model_path = os.path.abspath(
                os.path.join(os.curdir, './builds/' + timestamp))
        model_path += '-reorder-full-rd-elmo_checkpoint_epoch-{epoch:02d}.hdf5'

        saver = keras.callbacks.ModelCheckpoint(model_path,
                                                monitor='val_loss', verbose=verbose, save_best_only=True)

        keras_model.fit_generator(generator_training, steps_per_epoch=300,
                                  epochs=self.config.n_epochs,
                                  verbose=verbose,
                                  validation_data=generator_dev,
                                  validation_steps=len(test_set) / self.config.batch_size,
                                  callbacks=[tensorboard, saver])

    def eval(self):
        pass

    def build_graph(self, sess):
        if self.config.debug:
            print('Importing Elmo/USE module...')
        if self.config.hub.is_set("cache_dir"):
            os.environ['TFHUB_CACHE_DIR'] = self.config.hub.cache_dir

        embedding_types = {
            "elmo": {
                "name": "elmo",
                "url": "https://tfhub.dev/google/elmo/1",
                "size": 1024
            },
            "use": {
                "name": "use",
                "url": "https://tfhub.dev/google/universal-sentence-encoder/1",
                "size": 512
            }
        }[self.config.embedding_type]
        elmo_model = hub.Module(embedding_types['url'], trainable=True)

        if self.config.debug:
            print('Imported.')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        elmo_embeddings = keras.layers.Lambda(ElmoEmbedding(elmo_model, embedding_types), output_shape=(embedding_types['size'],))

        sentence1 = keras.layers.Input(shape=(1,), dtype="string")
        sentence2 = keras.layers.Input(shape=(1,), dtype="string")
        sentence3 = keras.layers.Input(shape=(1,), dtype="string")
        sentence4 = keras.layers.Input(shape=(1,), dtype="string")
        sentence5 = keras.layers.Input(shape=(1,), dtype="string")

        sentence1_emb = elmo_embeddings(sentence1)
        sentence2_emb = elmo_embeddings(sentence2)
        sentence3_emb = elmo_embeddings(sentence3)
        sentence4_emb = elmo_embeddings(sentence4)
        sentence5_emb = elmo_embeddings(sentence5)

        sentences = keras.layers.concatenate(
                [sentence1_emb, sentence2_emb, sentence3_emb, sentence4_emb, sentence5_emb])
        output = keras.layers.Dense(1000, activation='relu')(sentences)
        output = keras.layers.Dropout(0.2)(output)
        output = keras.layers.Dense(500, activation='relu')(output)
        output = keras.layers.Dropout(0.2)(output)
        output = keras.layers.Dense(1, activation='sigmoid')(output)

        # Model
        entailment_model = keras.models.Model(inputs=[sentence1, sentence2, sentence3, sentence4, sentence5],
                                              outputs=output)
        entailment_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
        return entailment_model
