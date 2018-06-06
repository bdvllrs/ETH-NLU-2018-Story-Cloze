"""
Credits to
- Matteo Pagliardini, Prakhar Gupta, Martin Jaggi,
    Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features NAACL 2018,
    https://arxiv.org/abs/1703.02507
- A large annotated corpus for learning natural language inference,
    _Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning_,
    https://nlp.stanford.edu/pubs/snli_paper.pdf.
"""
import datetime
import os
import random

import keras
import numpy as np
from utils import SNLIDataloaderPairs
from nltk import word_tokenize
from utils import Dataloader
from scripts import DefaultScript


class Script(DefaultScript):

    slug = 'entailment_v5'

    def train(self):
        main(self.config)

    # def test(self):
    #     testing_set = Dataloader(self.config, testing_data=True)
    #     testing_set.load_dataset('data/test.bin')
    #
    #     testing_set.load_vocab('data/default.voc', self.config.vocab_size)
    #     test(self.config, testing_set)


class Preprocess:
    def __init__(self, sent2vec):
        self.sent2vec = sent2vec

    def __call__(self, line):
        # label = [entailment, neutral, contradiction]
        sentence1 = list(self.sent2vec.embed_sentence(' '.join(word_tokenize(line['sentence1']))))
        sentence2 = list(self.sent2vec.embed_sentence(' '.join(word_tokenize(line['sentence2']))))
        return sentence1, sentence2


def output_fn(_, batch):
    labels = []
    sentence1 = []
    sentence2 = []
    sentence3 = []
    for b in range(len(batch)):
        sentence1.append(batch[b][0][0])
        if random.random() > 0.5:
            sentence2.append(batch[b][0][1])
            sentence3.append(batch[b][1][1])
            labels.append(0)
        else:
            sentence2.append(batch[b][1][1])
            sentence3.append(batch[b][0][1])
            labels.append(1)
    return [np.array(sentence1), np.array(sentence2), np.array(sentence3)], np.array(labels)


class OutputFnTest:
    def __init__(self, sent2vec, config):
        self.sent2vec = sent2vec
        self.config = config

    def __call__(self, data):
        batch = data.batch
        sentence_batch = []
        ending_1 = []
        ending_2 = []
        for b in range(len(batch)):
            # sentence = batch[b][0] + ' ' + batch[b][1] + ' ' + batch[b][2] + ' ' + batch[b][3]
            sentence = " ".join(batch[b][3])
            sentence_batch.append(self.sent2vec.embed_sentence(sentence))
            ending_1.append(self.sent2vec.embed_sentence(" ".join(batch[b][4])))
            ending_2.append(self.sent2vec.embed_sentence(" ".join(batch[b][5])))
        correct_ending = data.label
        label = np.array(correct_ending) - 1
        return [np.array(sentence_batch), np.array(ending_1), np.array(ending_2)], np.array(label)


def model(config):
    dense_layer_1 = keras.layers.Dense(2048, activation='relu')
    dense_layer_2 = keras.layers.Dense(1024, activation='relu')
    dense_layer_3 = keras.layers.Dense(512, activation='relu')
    dense_layer_4 = keras.layers.Dense(1, activation='sigmoid')

    sentence_1 = keras.layers.Input(shape=(config.sent2vec.embedding_size,))
    sentence_2 = keras.layers.Input(shape=(config.sent2vec.embedding_size,))
    sentence_3 = keras.layers.Input(shape=(config.sent2vec.embedding_size,))
    # Graph
    inputs = keras.layers.Concatenate()([sentence_1, sentence_2, sentence_3])
    # inputs = sentiments
    output = keras.layers.BatchNormalization()(keras.layers.Dropout(0.3)(dense_layer_1(inputs)))
    output = keras.layers.BatchNormalization()(keras.layers.Dropout(0.3)(dense_layer_2(output)))
    output = keras.layers.BatchNormalization()(keras.layers.Dropout(0.3)(dense_layer_3(output)))
    output = dense_layer_4(output)

    # Model
    model = keras.models.Model(inputs=[sentence_1, sentence_2, sentence_3], outputs=[output])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    return model


def main(config):
    import sent2vec
    assert config.sent2vec.model is not None, "Please add sent2vec_model config value."
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(config.sent2vec.model)

    preprocess_fn = Preprocess(sent2vec_model)

    output_fn_test = OutputFnTest(sent2vec_model, config)

    train_set = SNLIDataloaderPairs('data/snli_1.0/snli_1.0_train.jsonl')
    train_set.set_preprocess_fn(preprocess_fn)
    train_set.set_output_fn(output_fn)

    test_set = Dataloader(config, 'data/test_stories.csv', testing_data=True)
    test_set.load_dataset('data/test.bin')
    test_set.load_vocab('./data/default.voc', config.vocab_size)
    test_set.set_output_fn(output_fn_test)
    # dev_set = SNLIDataloader('data/snli_1.0/snli_1.0_dev.jsonl')
    # dev_set.set_preprocess_fn(preprocess_fn)
    # dev_set.set_output_fn(output_fn)
    # test_set = SNLIDataloader('data/snli_1.0/snli_1.0_test.jsonl')

    generator_training = train_set.get_batch(config.batch_size, config.n_epochs)
    generator_dev = test_set.get_batch(config.batch_size, config.n_epochs)
    print(next(generator_training))

    keras_model = model(config)

    verbose = 0 if not config.debug else 1
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Callbacks
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/' + timestamp + '-entailmentv5/', histogram_freq=0,
                                              batch_size=config.batch_size,
                                              write_graph=False,
                                              write_grads=True)

    model_path = os.path.abspath(
        os.path.join(os.curdir, './builds/' + timestamp))
    model_path += '-entailmentv5_checkpoint_epoch-{epoch:02d}.hdf5'

    saver = keras.callbacks.ModelCheckpoint(model_path,
                                            monitor='val_loss', verbose=verbose, save_best_only=True)

    keras_model.fit_generator(generator_training, steps_per_epoch=1,
                              epochs=config.n_epochs,
                              verbose=verbose,
                              validation_data=generator_dev,
                              validation_steps=len(test_set) / config.batch_size,
                              callbacks=[tensorboard, saver])


# def test(config, testing_set):
#     import sent2vec
#     assert config.sent2vec.model is not None, "Please add sent2vec_model config value."
#     sent2vec_model = sent2vec.Sent2vecModel()
#     sent2vec_model.load_model(config.sent2vec.model)
#
#     preprocess_fn = PreprocessTest(sent2vec_model)
#     testing_set.set_preprocess_fn(preprocess_fn)
#
#     output_fn_test = OutputFnTest(sent2vec_model, config)
#
#     testing_set.set_output_fn(output_fn_test)
#
#     generator_testing = testing_set.get_batch(config.batch_size, config.n_epochs, random=True)
#
#     keras_model = keras.models.load_model(
#         './builds/leonhard/2018-05-19 22:33:08-entailmentv2_checkpoint_epoch-1810.hdf5')
#
#     verbose = 0 if not config.debug else 1
#
#     # test_batch = next(generator_testing)
#     loss = keras_model.evaluate_generator(generator_testing, steps=len(testing_set) / config.batch_size,
#                                           verbose=verbose)
#     print(loss)
