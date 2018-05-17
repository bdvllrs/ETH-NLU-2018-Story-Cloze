import random

import datetime
import tensorflow as tf
import keras
from models import SentenceEmbedding
from utils import train_test
import numpy as np
import sent2vec


# class Preprocess:
#     def __init__(self, sent2vec_mode):
#         self.sent2vec_model = sent2vec_model
#
#     def __call__(self, word_to_index, sentence):
#         # Get sentence level embedding with sent2vec
#         sentence = self.sent2vec_model.embed_sentence(' '.join(sentence))
#         return sentence


class Preprocess:
    def __init__(self, config):
        self.config = config

    def __call__(self, word_to_index, sentence):
        # Get sentence level embedding with sent2vec
        sentence = list(
            map(lambda word: (word_to_index[word] if word in word_to_index.keys() else word_to_index['<unk>']),
                sentence))
        sentence += [word_to_index['<pad>']] * (self.config.max_size - len(sentence))
        return sentence


def mix_batches(right_ending, wrong_ending, sentiments, bad_ending_sentiment):
    batch1 = np.zeros(right_ending.shape)
    batch2 = np.zeros(wrong_ending.shape)
    label1 = np.zeros((len(right_ending), 2))
    label2 = np.zeros((len(right_ending), 2))
    sent1 = sentiments[:]
    sent2 = sentiments[:]
    for k in range(len(right_ending)):
        if random.random() > 0.5:
            batch1[k] = right_ending[k]
            label1[k][0] = 1
            label1[k][1] = 0
            batch2[k] = wrong_ending[k]
            label2[k][0] = 0
            label2[k][1] = 1
            sent2[k, -2:] = bad_ending_sentiment[k]
        else:
            batch2[k] = right_ending[k]
            label2[k][0] = 1
            label2[k][1] = 0
            batch1[k] = wrong_ending[k]
            label1[k][0] = 0
            label1[k][1] = 1
            sent1[k, -2:] = bad_ending_sentiment[k]
    return batch1, label1, batch2, label2, sent1, sent2


def main(config, training_set, testing_set):
    assert config.sent2vec.model is not None, "Please add sent2vec_model config value."
    # model = sent2vec.Sent2vecModel()
    # model.load_model(config.sent2vec.model)
    # preprocess = Preprocess(model)
    preprocess = Preprocess(config)
    training_set.set_preprocess_fn(preprocess)
    testing_set.set_preprocess_fn(preprocess)

    sentence_embedding = SentenceEmbedding(config)
    sentence_embedding()
    sentence_embedding.optimize()

    keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=config.batch_size, write_graph=False,
                                write_grads=True)

    model_last_sentence = keras.layers.Input(shape=(config.max_size,))
    model_ending = keras.layers.Input(shape=(config.max_size,))
    model_sentiment = keras.layers.Input(shape=(10,))
    model_embedding = keras.layers.Embedding(config.vocab_size, config.embedding_size)
    model_last_sentence_embedded = model_embedding(model_last_sentence)
    model_ending_embedded = model_embedding(model_ending)
    model_flatten = keras.layers.Flatten()
    output = keras.layers.Concatenate()(
        [model_flatten(model_last_sentence_embedded), model_flatten(model_ending_embedded),
         model_sentiment])
    output = keras.layers.Dense(units=500, activation='relu')(output)
    output = keras.layers.Dense(units=50, activation='relu')(output)
    output = keras.layers.Dense(units=2, activation='sigmoid')(output)

    model = keras.models.Model(inputs=[model_last_sentence, model_ending, model_sentiment], outputs=output)

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    train_writer = tf.summary.FileWriter('./logs/' + timestamp + '/train/')

    for epoch in range(config.n_epochs):
        if config.debug:
            print("Epoch", epoch)

        for k in range(0, len(training_set), config.batch_size):
            batch, sentiments = training_set.get(k, config.batch_size, random=True, with_sentiments=True)
            # Get random endings to train bad endings
            bad_ending, bad_ending_sentiments = training_set.get(k, config.batch_size, random=True,
                                                                 with_sentiments=True)
            bad_ending = bad_ending[:, 4, :]
            bad_ending_sentiments = bad_ending_sentiments[:, 8:10]
            last_sentences, right_ending = batch[:, 3, :], batch[:, 4, :]
            batch1, label1, batch2, label2, sent1, sent2 = mix_batches(right_ending, bad_ending, sentiments,
                                                                       bad_ending_sentiments)
            # input_batch1 = np.concatenate((last_sentences + batch1, sent1), axis=1)
            # input_batch2 = np.concatenate((last_sentences + batch2, sent2), axis=1)

            loss, acc = model.train_on_batch([last_sentences, batch1, sent1], label1)
            loss2, acc2 = model.train_on_batch([last_sentences, batch2, sent2], label2)
            loss_summary = tf.Summary()
            acc_summary = tf.Summary()
            loss_summary.value.add(tag='loss', simple_value=loss)
            loss_summary.value.add(tag='loss', simple_value=loss2)
            acc_summary.value.add(tag='acc', simple_value=acc)
            acc_summary.value.add(tag='acc', simple_value=acc2)
            train_writer.add_summary(loss_summary, epoch * len(training_set) + k)
            train_writer.add_summary(acc_summary, epoch * len(training_set) + k)
