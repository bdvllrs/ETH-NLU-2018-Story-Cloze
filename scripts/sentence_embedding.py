import random

import tensorflow as tf
import numpy as np
from models import SentenceEmbedding
from utils import train_test
import sent2vec


class Preprocess:
    def __init__(self, sent2vec_model):
        self.sent2vec_model = sent2vec_model

    def __call__(self, word_to_index, sentence):
        # Get sentence level embedding with sent2vec
        sentence = self.sent2vec_model.embed_sentence(' '.join(sentence))
        return sentence


def mix_batches(right_ending, wrong_ending, sentiments, bad_ending_sentiment):
    batch1 = np.zeros(right_ending.shape)
    batch2 = np.zeros(wrong_ending.shape)
    label1 = np.zeros(len(right_ending))
    label2 = np.zeros(len(right_ending))
    sent1 = sentiments[:]
    sent2 = sentiments[:]
    for k in range(len(right_ending)):
        if random.random() > 0.5:
            batch1[k] = right_ending[k]
            label1[k] = 1
            batch2[k] = wrong_ending[k]
            label2[k] = 0
            sent2[k, -2:] = bad_ending_sentiment[k]
        else:
            batch2[k] = right_ending[k]
            label2[k] = 1
            batch1[k] = wrong_ending[k]
            label1[k] = 0
            sent1[k, -2:] = bad_ending_sentiment[k]
    return batch1, label1, batch2, label2, sent1, sent2


def test_fn(config, testing_set, sess, epoch, k):
    """
    Function executed for each batch for testing
    :param config:
    :param testing_set:
    :param sess:
    :param epoch:
    :param k:
    """
    batch_endings1, batch_endings2, correct_ending, sent1, sent2 = testing_set.get(k, config.batch_size,
                                                                                   random=True, with_sentiments=True)
    sentence1, ending1 = batch_endings1[:, 3, :], batch_endings1[:, 4, :]
    sentence2, ending2 = batch_endings2[:, 3, :], batch_endings2[:, 4, :]
    output = [None, None]
    output[0] = sess.run(
        'sentence_embedding/output:0',
        {'sentence_embedding/last-sentence:0': sentence1,
         'sentence_embedding/sentiment:0': sent1,
         'sentence_embedding/ending:0': ending1})
    output[1] = sess.run(
        'sentence_embedding/output:0',
        {'sentence_embedding/last-sentence:0': sentence2,
         'sentence_embedding/sentiment:0': sent2,
         'sentence_embedding/ending:0': ending2})
    success = 0
    for b in range(config.batch_size):
        if output[correct_ending[b]][b] > output[1 - correct_ending[b]][b]:
            success += 1
    success = 0
    return config.batch_size, success


def train_fn(config, training_set, sess, epoch, k, summary_op, train_writer):
    """
    Function executed for each batch for training
    :param config:
    :param training_set:
    :param sess:
    :param epoch:
    :param k:
    :param summary_op:
    :param train_writer:
    """
    batch, sentiments = training_set.get(k, config.batch_size, random=True, with_sentiments=True)
    # Get random endings to train bad endings

    bad_ending, bad_ending_sentiments = training_set.get(k+1, config.batch_size, random=True, with_sentiments=True)
    bad_ending = bad_ending[:, 4, :]
    bad_ending_sentiments = bad_ending_sentiments[:, 8:10]
    last_sentences, right_ending = batch[:, 3, :], batch[:, 4, :]
    batch1, label1, batch2, label2, sent1, sent2 = mix_batches(right_ending, bad_ending, sentiments,
                                                               bad_ending_sentiments)
    _, summary = sess.run(
        ['sentence_embedding/optimize/optimizer', summary_op],
        {'sentence_embedding/last-sentence:0': last_sentences,
         'sentence_embedding/ending:0': batch1,
         'sentence_embedding/sentiment:0': sent1,
         'sentence_embedding/optimize/label:0': label1})  # High label for this one as it is a right ending
    _, summary2 = sess.run(
        ['sentence_embedding/optimize/optimizer', summary_op],
        {'sentence_embedding/last-sentence:0': last_sentences,
         'sentence_embedding/ending:0': batch2,
         'sentence_embedding/sentiment:0': sent2,
         'sentence_embedding/optimize/label:0': label2})  # High label for this one as it is a right ending
    train_writer.add_summary(summary, epoch * len(training_set) + k)
    train_writer.add_summary(summary2, epoch * len(training_set) + k + 0.5)


def main(config, training_set, testing_set):
    assert config.sent2vec.model is not None, "Please add sent2vec_model config value."
    model = sent2vec.Sent2vecModel()
    model.load_model(config.sent2vec.model)
    preprocess = Preprocess(model)
    training_set.set_preprocess_fn(preprocess)
    testing_set.set_preprocess_fn(preprocess)

    sentence_embedding = SentenceEmbedding(config)
    sentence_embedding()
    sentence_embedding.optimize()

    tf.summary.scalar("cross_entropy", sentence_embedding.cross_entropy)

    train_test(config, training_set, testing_set, test_fn, train_fn)
