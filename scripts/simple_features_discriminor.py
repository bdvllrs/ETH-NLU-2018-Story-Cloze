import datetime
import os
import random

import numpy as np
import tensorflow as tf

import keras
from keras.layers import Embedding, Flatten, Dense, Dropout, Input, concatenate

from scripts import DefaultScript
from utils import PPDataloader


def preprocess_fn(word_to_index, sentences):
    sentiments = []
    topics = []
    label = None
    for sentence in sentences:
        if type(sentence) == int:
            label = sentence
        else:
            sentiments.append(sentence[0])
            topic = sentence[1][1:]
            topic_list = []
            for t in topic:
                topic_list.append(word_to_index[t] if t in word_to_index.keys() else word_to_index['<unk>'])
                # Always send the same order for the topics
                topic_list = sorted(topic_list)
            if len(topic_list) < 5:
                topic_list += [word_to_index['<pad>'] for _ in range(5 - len(topic_list))]
            elif len(topic_list) > 5:
                topic_list = topic_list[:5]
            topics.append(topic_list)
    return sentiments, topics, label


def output_fn(_, batch):
    sentiments = []
    sentences_1 = []
    sentences_2 = []
    sentences_3 = []
    sentences_4 = []
    endings_1 = []
    endings_2 = []
    labels = []
    for b in batch:
        sentiment, topics, label = b
        sentiments.append(sentiment)
        sentences_1.append(topics[0])
        sentences_2.append(topics[1])
        sentences_3.append(topics[2])
        sentences_4.append(topics[3])
        endings_1.append(topics[4])
        endings_2.append(topics[5])
        labels.append(label)  # 0 if ending_1 is correct, 1 if ending_2 is correct
    return [np.array(sentences_1), np.array(sentences_2), np.array(sentences_3), np.array(sentences_4),
            np.array(endings_1), np.array(endings_2), np.array(sentiments)], np.array(labels)


class Script(DefaultScript):
    slug = 'simple_features_discriminator'

    def train(self):
        self.generator_model = keras.models.load_model(
                './builds/leonhard/2018-06-07 17:11:04-simple_features_generator_epoch-03.hdf5')

        self.graph = tf.get_default_graph()

        train_set = PPDataloader('./data/train_features.pkl')
        train_set.load_vocab('./data/dev_topics.pkl', size_percent=0.8)
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
        tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/' + timestamp + '-simple_features_discriminator/',
                                                  histogram_freq=0,
                                                  batch_size=self.config.batch_size,
                                                  write_graph=False,
                                                  write_grads=True)

        model_path = os.path.abspath(
                os.path.join(os.curdir, './builds/' + timestamp))
        model_path += '-simple_features_discriminator_epoch-{epoch:02d}.hdf5'

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
        sentence_1 = Input((5,))
        sentence_2 = Input((5,))
        sentence_3 = Input((5,))
        sentence_4 = Input((5,))
        ending_1 = Input((5,))
        ending_2 = Input((5,))
        sentiment = Input((6,))

        # Layers
        embedding_layer = Embedding(self.config.vocab_size, 16, input_length=5)
        flatten = Flatten()
        layer_1 = Dense(64, activation="relu", name="discriminator_layer_1")
        layer_2 = Dense(1, activation="sigmoid", name="discriminator_layer_2")

        sentence_1_embedded = flatten(embedding_layer(sentence_1))
        sentence_2_embedded = flatten(embedding_layer(sentence_2))
        sentence_3_embedded = flatten(embedding_layer(sentence_3))
        sentence_4_embedded = flatten(embedding_layer(sentence_4))
        ending_1_embedded = flatten(embedding_layer(ending_1))
        ending_2_embedded = flatten(embedding_layer(ending_2))

        # Build graph
        features = concatenate(
                [sentence_1_embedded, sentence_2_embedded, sentence_3_embedded, sentence_4_embedded, ending_1_embedded,
                 ending_2_embedded, sentiment])  # of size 1542

        output = Dropout(0.5)(layer_1(features))
        output = layer_2(output)

        model = keras.models.Model([sentence_1, sentence_2, sentence_3, sentence_4, ending_1, ending_2, sentiment],
                                   output)
        model.compile(keras.optimizers.Adam(lr=0.0005), 'binary_crossentropy', ['accuracy'])
        return model

    def output_fn_train(self, _, batch):
        sentiments = []
        sentences_1 = []
        sentences_2 = []
        sentences_3 = []
        sentences_4 = []
        endings_1 = []
        endings_2 = []
        labels = []
        for b in batch:
            sentiment, topics, label = b
            sentiments.append(sentiment)
            sentences_1.append(topics[0])
            sentences_2.append(topics[1])
            sentences_3.append(topics[2])
            sentences_4.append(topics[3])
            with self.graph.as_default():
                generated_output = self.generator_model.predict(
                        [np.array([topics[0]]), np.array([topics[1]]), np.array([topics[2]]), np.array([topics[3]]),
                         np.array([topics[4]]), np.array([sentiment])], batch_size=1)
            generated_output = np.argmax(generated_output, axis=2)
            print(generated_output)
            if random.random() > 0.5:
                endings_1.append(topics[4])
                endings_2.append(generated_output)
                labels.append(0)
            else:
                endings_1.append(generated_output)
                endings_2.append(topics[4])
                labels.append(1)
        return [np.array(sentences_1), np.array(sentences_2), np.array(sentences_3), np.array(sentences_4),
                np.array(endings_1), np.array(endings_2), np.array(sentiments)], np.array(labels)
