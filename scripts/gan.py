"""
Credits to
- A large annotated corpus for learning natural language inference,
    _Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning_,
    https://nlp.stanford.edu/pubs/snli_paper.pdf.
- AllenNLP for Elmo Embeddings: Deep contextualized word representations
    _Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer_,
    https://arxiv.org/abs/1802.05365.
- Jacob Zweig for Elmo embedding import code from
https://towardsdatascience.com/elmo-embeddings-in-keras-with-tensorflow-hub-7eb6f0145440.
"""
import datetime
import os
import random

import keras
from keras import backend as K
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from utils import SNLIDataloader, Dataloader
from nltk import word_tokenize
from scripts import DefaultScript


class Script(DefaultScript):
    slug = 'gan'

    def train(self):
        main(self.config)


class OutputFN:
    def __init__(self, elmo_model, type_translation_model, graph):
        self.elmo_model = elmo_model
        self.type_translation_model = type_translation_model
        self.graph = graph

    def __call__(self, data):
        batch = np.array(data.batch)
        ref_sentences = []
        input_sentences = []
        label = []
        for b in batch:
            sentence = " ".join(b[0]) + " "
            sentence += " ".join(b[1]) + " "
            sentence += " ".join(b[2]) + " "
            sentence += " ".join(b[3])
            # Concatenate the story for only one sentence
            ref_sentences.append(sentence)
            input_sentences.append(" ".join(b[4]))
        ref_sentences, input_sentences = np.array(ref_sentences, dtype=object), np.array(input_sentences, dtype=object)
        with self.graph.as_default():
            # Get the prediction of the negative sentence by our type transfert model
            negative_sentence_emb = self.type_translation_model.predict([ref_sentences, input_sentences],
                                                                        batch_size=len(batch))
            # Get the elmo embeddings for the input sentences and ref sentences (stories)
            input_sentences_emb = self.elmo_model.predict(input_sentences, batch_size=len(batch))
            ref_sentences_emb = self.elmo_model.predict(ref_sentences, batch_size=len(batch))
        labels = []
        output_sentences = []
        for b in range(len(batch)):
            # Randomly choose one or the other
            if random.random() > 0.5:
                output_sentences.append(negative_sentence_emb[b])
                labels.append(0)
            else:
                output_sentences.append(input_sentences_emb[b])
                labels.append(1)
        return [ref_sentences_emb, np.array(output_sentences)], np.array(labels)


class OutputFNTest:
    def __init__(self, elmo_model, type_translation_model, graph):
        self.elmo_model = elmo_model
        self.type_translation_model = type_translation_model
        self.graph = graph

    def __call__(self, data):
        batch = np.array(data.batch)
        ref_sentences = []
        input_sentences = []
        label = []
        for b in batch:
            sentence = " ".join(b[0]) + " "
            sentence += " ".join(b[1]) + " "
            sentence += " ".join(b[2]) + " "
            sentence += " ".join(b[3]) + " "
            if random.random() > 0.5:
                input_sentences.append(" ".join(b[4]))
                label.append(2 - int(b[6][0]))
            else:
                input_sentences.append(" ".join(b[5]))
                label.append(int(b[6][0]) - 1)
            ref_sentences.append(sentence)
        ref_sentences, input_sentences = np.array(ref_sentences, dtype=object), np.array(input_sentences, dtype=object)
        with self.graph.as_default():
            # Get the elmo embeddings for the input sentences and ref sentences (stories)
            input_sentences_emb = self.elmo_model.predict(input_sentences, batch_size=len(batch))
            ref_sentences_emb = self.elmo_model.predict(ref_sentences, batch_size=len(batch))
        return [ref_sentences_emb, input_sentences_emb], np.array(label)


class ElmoEmbedding:
    def __init__(self, elmo_model):
        self.elmo_model = elmo_model
        self.__name__ = "elmo_embeddings"

    def __call__(self, x):
        return self.elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)[
            "default"]


def get_elmo_embedding(elmo_fn):
    elmo_embeddings = keras.layers.Lambda(elmo_fn, output_shape=(1024,))
    sentence = keras.layers.Input(shape=(1,), dtype="string")
    sentence_emb = elmo_embeddings(sentence)
    model = keras.models.Model(inputs=sentence, outputs=sentence_emb)
    return model


def discriminator():
    sentence1 = keras.layers.Input(shape=(1024,))
    sentence2 = keras.layers.Input(shape=(1024,))

    sentences = keras.layers.concatenate([sentence1, sentence2])
    output = keras.layers.Dense(1000, activation='relu')(sentences)
    output = keras.layers.Dropout(0.2)(output)
    output = keras.layers.Dense(500, activation='relu')(output)
    output = keras.layers.Dropout(0.2)(output)
    output = keras.layers.Dense(1, activation='sigmoid')(output)

    # Model
    entailment_model = keras.models.Model(inputs=[sentence1, sentence2], outputs=output)
    entailment_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    return entailment_model


def main(config):
    # Initialize tensorflow session
    sess = tf.Session()
    K.set_session(sess)  # Set to keras backend

    if config.debug:
        print('Importing Elmo module...')
    if config.hub.is_set("cache_dir"):
        os.environ['TFHUB_CACHE_DIR'] = config.hub.cache_dir

    elmo_model = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
    if config.debug:
        print('Imported.')

    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    graph = tf.get_default_graph()

    elmo_emb_fn = ElmoEmbedding(elmo_model)

    elmo_model_emb = get_elmo_embedding(elmo_emb_fn)

    generator_model = keras.models.load_model(
            config.type_translation_model, {
                'elmo_embeddings': elmo_emb_fn
            })

    output_fn = OutputFN(elmo_model_emb, generator_model, graph)
    output_fn_test = OutputFNTest(elmo_model_emb, generator_model, graph)
    train_set = Dataloader(config, 'data/train_stories.csv')
    # test_set.set_preprocess_fn(preprocess_fn)
    train_set.load_dataset('data/train.bin')
    train_set.load_vocab('./data/default.voc', config.vocab_size)
    train_set.set_output_fn(output_fn)

    test_set = Dataloader(config, 'data/test_stories.csv', testing_data=True)
    # test_set.set_preprocess_fn(preprocess_fn)
    test_set.load_dataset('data/test.bin')
    test_set.load_vocab('./data/default.voc', config.vocab_size)
    test_set.set_output_fn(output_fn_test)

    generator_training = train_set.get_batch(config.batch_size, config.n_epochs)
    generator_test = test_set.get_batch(config.batch_size, config.n_epochs)

    # print(next(generator_training))

    discriminator_model = discriminator()

    # verbose = 0 if not config.debug else 1
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # # Callbacks
    # tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/' + timestamp + '-entailmentv4/', histogram_freq=0,
    #                                           batch_size=config.batch_size,
    #                                           write_graph=False,
    #                                           write_grads=True)

    model_path = os.path.abspath(
            os.path.join(os.curdir, './builds/' + timestamp))
    model_path += '-entailmentv4_checkpoint_epoch-{epoch:02d}.hdf5'

    # saver = keras.callbacks.ModelCheckpoint(model_path,
    #                                         monitor='val_loss', verbose=verbose, save_best_only=True)
    #
    # discriminator_model.fit_generator(generator_training, steps_per_epoch=1000,
    #                           epochs=config.n_epochs,
    #                           verbose=verbose,
    #                           validation_data=generator_test,
    #                           validation_steps=len(test_set) / config.batch_size,
    #                           callbacks=[tensorboard, saver])
    for epoch in range(config.n_epochs):
        # Training
        for batch in generator_training:
            pass
            # Train discriminator
