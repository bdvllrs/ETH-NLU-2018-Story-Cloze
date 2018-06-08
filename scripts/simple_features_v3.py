import datetime
import os
import random

import numpy as np

import keras
import tensorflow as tf
from keras.layers import Embedding, Flatten, Dense, GRU, Dropout, Input, concatenate, Reshape
from keras.utils import to_categorical

from scripts import DefaultScript
from utils import PPDataloader


def preprocess_fn(_, sentences):
    sentiments = []
    label = None
    for sentence in sentences:
        if type(sentence) == int:
            label = sentence
        else:
            sentiments.append(sentence[0])
    return sentiments, label


def output_fn(_, batch):
    sentiments = []
    labels = []
    for b in batch:
        sentiment, label = b
        sentiments.append(sentiment)
        labels.append(label)  # 0 if ending_1 is correct, 1 if ending_2 is correct
    return np.array(sentiments), np.array(labels)


class Script(DefaultScript):
    slug = 'simple_features_v3'

    def train(self):
        self.generator_model = keras.models.load_model(
                './builds/2018-06-07 23:43:46-simple_features_epoch-14.hdf5')

        self.graph = tf.get_default_graph()

        train_set = PPDataloader('./data/train_features.pkl')
        train_set.load_vocab('./data/train_topics.pkl', size_percent=0.8)
        train_set.set_preprocess_fn(preprocess_fn)
        train_set.set_output_fn(self.output_fn_train)

        test_set = PPDataloader('./data/test_features.pkl')
        test_set.load_vocab('./data/dev_topics.pkl', size_percent=0.8)
        test_set.set_preprocess_fn(preprocess_fn)
        test_set.set_output_fn(output_fn)

        self.config.set('vocab_size', len(train_set.index_to_word))

        model = self.build_classifier_graph()

        train_generator = train_set.get_batch(self.config.batch_size, self.config.n_epochs)
        test_generator = test_set.get_batch(self.config.batch_size, self.config.n_epochs)

        verbose = 0 if not self.config.debug else 1
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Callbacks
        tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/' + timestamp + '-simple_features_v3/',
                                                  histogram_freq=0,
                                                  batch_size=self.config.batch_size,
                                                  write_graph=False,
                                                  write_grads=True)

        model_path = os.path.abspath(
                os.path.join(os.curdir, './builds/' + timestamp))
        model_path += '-simple_features_v3_epoch-{epoch:02d}.hdf5'

        saver = keras.callbacks.ModelCheckpoint(model_path,
                                                monitor='val_loss', verbose=verbose, save_best_only=True)

        model.fit_generator(train_generator, steps_per_epoch=3000,
                            epochs=self.config.n_epochs,
                            verbose=verbose,
                            validation_data=test_generator,
                            validation_steps=len(test_set) / self.config.batch_size,
                            callbacks=[tensorboard, saver])
        # model = self.build_seq_to_seq_graph()
        # x, y = dev_set.get(1, 2)
        # print(model.train_on_batch(x, y))

    def build_classifier_graph(self):
        sentiment = Input((6,))

        # Layers
        layer_1 = Dense(8, activation="relu")
        layer_2 = Dense(4, activation="relu")
        layer_3 = Dense(1, activation="sigmoid")

        output = Dropout(0.1)(layer_1(sentiment))
        output = Dropout(0.1)(layer_2(output))
        output = layer_3(output)

        model = keras.models.Model(sentiment, output)
        model.compile(keras.optimizers.Adam(lr=0.001), 'binary_crossentropy', ['accuracy'])
        return model

    def output_fn_train(self, _, batch):
        sentiments = []
        labels = []
        for b in batch:
            sentiment, label = b
            with self.graph.as_default():
                generated_output = self.generator_model.predict(np.array([sentiment]), batch_size=1)
            generated_output = np.argmax(generated_output, axis=1)
            sentiment_end = sentiment[4]
            sentiment_end_2 = 1 - generated_output[0]
            sentiment_refs = sentiment[:4]
            if random.random() > 0.5:
                sentiment = sentiment_refs + [sentiment_end] + [sentiment_end_2]
                labels.append(0)
            else:
                sentiment = sentiment_refs + [sentiment_end_2] + [sentiment_end]
                labels.append(1)
            sentiments.append(sentiment)
        return np.array(sentiments), np.array(labels)
