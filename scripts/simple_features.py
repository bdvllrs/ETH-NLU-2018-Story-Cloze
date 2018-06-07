import datetime
import os

import numpy as np

import keras
from keras.layers import Embedding, Flatten, Dense, GRU, Dropout, Input, concatenate, Reshape
from keras.utils import to_categorical

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
    slug = 'simple_features'

    def train(self):
        train_set = PPDataloader('./data/dev_features.pkl')
        train_set.load_vocab('./data/dev_topics.pkl', size_percent=0.8)
        train_set.set_preprocess_fn(preprocess_fn)
        train_set.set_output_fn(output_fn)

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
        sentence_1 = Input((5,))
        sentence_2 = Input((5,))
        sentence_3 = Input((5,))
        sentence_4 = Input((5,))
        ending_1 = Input((5,))
        ending_2 = Input((5,))
        sentiment = Input((6,))

        # Layers
        embedding_layer = Embedding(self.config.vocab_size, 128, input_length=5)
        flatten = Flatten()
        layer_1 = Dense(2048, activation="relu")
        layer_2 = Dense(1024, activation="relu")
        layer_3 = Dense(128, activation="relu")
        layer_4 = Dense(1, activation="sigmoid")

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

        output = Dropout(0.2)(layer_1(features))
        output = Dropout(0.2)(layer_2(output))
        output = Dropout(0.2)(layer_3(output))
        output = layer_4(output)

        model = keras.models.Model([sentence_1, sentence_2, sentence_3, sentence_4, ending_1, ending_2, sentiment],
                                   output)
        model.compile('adam', 'binary_crossentropy', ['accuracy'])
        return model

    def build_seq_to_seq_graph(self):
        sentence_1 = Input((5,))
        sentence_2 = Input((5,))
        sentence_3 = Input((5,))
        sentence_4 = Input((5,))
        ending_1 = Input((5,))
        sentiments = Input((5,))

        # Layers
        embedding_layer = Embedding(self.config.vocab_size, 128, input_length=5)
        softmax = Dense(self.config.vocab_size, activation="softmax")

        encoder = GRU(512, return_sequences=True, return_state=True)
        decoder = GRU(512, return_sequences=True, return_state=True)

        sentence_1_embedded = embedding_layer(sentence_1)
        sentence_2_embedded = embedding_layer(sentence_2)
        sentence_3_embedded = embedding_layer(sentence_3)
        sentence_4_embedded = embedding_layer(sentence_4)
        ending_1_embedded = embedding_layer(ending_1)

        sentiments_reshaped = Reshape((5, 1))(sentiments)

        # Build graph
        features = concatenate(
                [sentence_1_embedded, sentence_2_embedded, sentence_3_embedded, sentence_4_embedded,
                 ending_1_embedded])  # of size 5 x 768
        features = concatenate([features, sentiments_reshaped], axis=2)

        encoder_outputs, encoder_state = encoder(features)
        decoder_outputs, _ = decoder(encoder_outputs, initial_state=encoder_state)
        decoder_outputs = softmax(decoder_outputs)

        model = keras.models.Model(
                [sentence_1, sentence_2, sentence_3, sentence_4, ending_1, sentiments], decoder_outputs)
        model.compile('rmsprop', 'categorical_crossentropy', ['accuracy'])
        return model

    def output_fn_seq2seq(self, _, batch):
        sentiments = []
        sentences_1 = []
        sentences_2 = []
        sentences_3 = []
        sentences_4 = []
        ringht_ending = []
        wrong_ending = []
        for b in batch:
            sentiment, topics, _ = b
            sentiments.append(sentiment[0:5])
            sentences_1.append(topics[0])
            sentences_2.append(topics[1])
            sentences_3.append(topics[2])
            sentences_4.append(topics[3])
            ringht_ending.append(topics[4])
            wrong_ending.append(topics[5])
        return [np.array(sentences_1), np.array(sentences_2), np.array(sentences_3), np.array(sentences_4),
                np.array(ringht_ending), np.array(sentiments)], to_categorical(np.array(wrong_ending),
                                                                               self.config.vocab_size)
