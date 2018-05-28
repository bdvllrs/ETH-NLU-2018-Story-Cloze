import datetime

import keras
from keras import backend as K
import os
import tensorflow_hub as hub
import tensorflow as tf
from scripts import DefaultScript
from utils import Dataloader


class ElmoEmbedding:
    def __init__(self, elmo_model):
        self.elmo_model = elmo_model
        self.__name__ = "elmo_embeddings"

    def __call__(self, x):
        return self.elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)[
            "elmo"]


def preprocess_fn(_, sentence):
    return ' '.join(sentence)


def output_fn(data):
    return data.batch


class Script(DefaultScript):

    slug = 'reorder_elmo'

    def train(self):
        train_set = Dataloader(self.config, 'data/train_stories.csv')
        test_set = Dataloader(self.config, 'data/test_stories.csv', testing_data=True)
        train_set.set_preprocess_fn(preprocess_fn)
        test_set.set_preprocess_fn(preprocess_fn)

        # generator_training = train_set.get_batch(self.config.batch_size, self.config.n_epochs)
        # generator_dev = test_set.get_batch(self.config.batch_size, self.config.n_epochs)
        #
        # # Initialize tensorflow session
        # sess = tf.Session()
        # K.set_session(sess)  # Set to keras backend
        #
        # keras_model = self.build_graph(sess)
        #
        # verbose = 0 if not self.config.debug else 1
        # timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # # Callbacks
        # tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/' + timestamp + '-reorder-elmo/', histogram_freq=0,
        #                                           batch_size=self.config.batch_size,
        #                                           write_graph=False,
        #                                           write_grads=True)
        #
        # model_path = os.path.abspath(
        #         os.path.join(os.curdir, './builds/' + timestamp))
        # model_path += '-reorder-elmo_checkpoint_epoch-{epoch:02d}.hdf5'
        #
        # saver = keras.callbacks.ModelCheckpoint(model_path,
        #                                         monitor='val_acc', verbose=verbose, save_best_only=True)
        #
        # keras_model.fit_generator(generator_training, steps_per_epoch=300,
        #                           epochs=self.config.n_epochs,
        #                           verbose=verbose,
        #                           validation_data=generator_dev,
        #                           validation_steps=len(test_set) / self.config.batch_size,
        #                           callbacks=[tensorboard, saver])

    def eval(self):
        pass

    def build_graph(self, sess):
        if self.config.debug:
            print('Importing Elmo module...')
        if self.config.hub.is_set("cache_dir"):
            os.environ['TFHUB_CACHE_DIR'] = self.config.hub.cache_dir

        elmo_model = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)

        if self.config.debug:
            print('Imported.')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        elmo_embeddings = ElmoEmbedding(elmo_model)

        # TODO

        model = keras.models.Model(inputs=[], outputs=[])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
        return model
