import datetime
import os

import numpy as np

import keras
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
    slug = 'simple_features_v2'

    def train(self):
        train_set = PPDataloader('./data/dev_features.pkl')
        train_set.load_vocab('./data/dev_topics.pkl', size_percent=0.8)
        train_set.set_preprocess_fn(preprocess_fn)
        train_set.set_output_fn(self.output_fn_seq2seq)

        test_set = PPDataloader('./data/test_features.pkl')
        test_set.load_vocab('./data/dev_topics.pkl', size_percent=0.8)
        test_set.set_preprocess_fn(preprocess_fn)
        test_set.set_output_fn(self.output_fn_seq2seq)

        self.config.set('vocab_size', len(train_set.index_to_word))

        model = self.build_seq_to_seq_graph()

        train_generator = train_set.get_batch(self.config.batch_size, self.config.n_epochs)
        test_generator = test_set.get_batch(self.config.batch_size, self.config.n_epochs)

        verbose = 0 if not self.config.debug else 1
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Callbacks
        tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/' + timestamp + '-simple_features/',
                                                  histogram_freq=0,
                                                  batch_size=self.config.batch_size,
                                                  write_graph=False,
                                                  write_grads=True)

        model_path = os.path.abspath(
                os.path.join(os.curdir, './builds/' + timestamp))
        model_path += '-simple_features_epoch-{epoch:02d}.hdf5'

        saver = keras.callbacks.ModelCheckpoint(model_path,
                                                monitor='val_loss', verbose=verbose, save_best_only=True)

        model.fit_generator(train_generator, steps_per_epoch=len(train_set) / self.config.batch_size,
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
        layer_2 = Dense(1, activation="sigmoid")

        output = Dropout(0.1)(layer_1(sentiment))
        output = layer_2(output)

        model = keras.models.Model(sentiment, output)
        model.compile(keras.optimizers.Adam(lr=0.0005), 'binary_crossentropy', ['accuracy'])
        return model

    def build_seq_to_seq_graph(self):
        sentiments = Input((5,))

        # Layers
        layer_1 = Dense(8, activation="relu")
        layer_2 = Dense(3, activation="softmax")

        # Build graph
        output = Dropout(0.2)(layer_1(sentiments))
        output = layer_2(output)

        model = keras.models.Model(sentiments, output)
        model.compile('adam', 'categorical_crossentropy', ['accuracy'])
        return model

    def output_fn_seq2seq(self, _, batch):
        sentiments = []
        labels = []
        for b in batch:
            sentiment, label = b
            s = sentiment[0:4]
            right_ending = sentiment[4]
            wrong_ending = sentiment[5]
            if label == 1:
                right_ending = sentiment[5]
                wrong_ending = sentiment[4]
            sentiments.append(s + [right_ending])
            if wrong_ending <= -1:
                labels.append([0, 0, 1])
            elif wrong_ending == 0:
                labels.append([0, 1, 0])
            elif wrong_ending >= 1:
                labels.append([1, 0, 0])
        return np.array(sentiments), np.array(labels)
