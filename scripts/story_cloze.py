"""
Credits to
- Matteo Pagliardini, Prakhar Gupta, Martin Jaggi,
    Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features NAACL 2018,
    https://arxiv.org/abs/1703.02507
- A large annotated corpus for learning natural language inference,
    Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning,
    https://nlp.stanford.edu/pubs/snli_paper.pdf.
"""
import os

import keras
import tensorflow as tf
import tensorflow_hub as hub
import keras.backend as K
import numpy as np

from utils import Dataloader
from scripts import DefaultScript


class Script(DefaultScript):
    slug = 'story_cloze'

    def eval(self):
        # Initialize tensorflow session
        sess = tf.Session()
        K.set_session(sess)  # Set to keras backend

        if self.config.debug:
            print('Importing Elmo module...')
        if self.config.hub.is_set("cache_dir"):
            os.environ['TFHUB_CACHE_DIR'] = self.config.hub.cache_dir

        elmo_model = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
        if self.config.debug:
            print('Imported.')

        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        graph = tf.get_default_graph()

        elmo_emb_fn = ElmoEmbedding(elmo_model)

        elmo_model_emb = get_elmo_embedding(elmo_emb_fn)

        type_translation_model = keras.models.load_model(self.config.type_translation_model)

        output_fn = OutputFN(elmo_model_emb, type_translation_model, graph)

        test_set = Dataloader(self.config, 'data/test_stories.csv', testing_data=True)
        # test_set.set_preprocess_fn(preprocess_fn)
        test_set.load_dataset('data/test.bin')
        test_set.load_vocab('./data/default.voc', self.config.vocab_size)
        test_set.set_output_fn(output_fn)

        generator_test = test_set.get_batch(self.config.batch_size, 1)
        accuracy = []
        for batch in generator_test:
            accuracy.append(batch)
            print(np.mean(accuracy))


class ElmoEmbedding:
    def __init__(self, elmo_model):
        self.elmo_model = elmo_model
        self.__name__ = "elmo_embeddings"

    def __call__(self, x):
        return self.elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)[
            "default"]


class OutputFN:
    def __init__(self, elmo_emb_model, type_translation_model, graph):
        self.type_translation_model = type_translation_model
        self.graph = graph
        self.elmo_model = elmo_emb_model

    def __call__(self, data):
        batch = np.array(data.batch)
        ref_sentences = []
        sentence1 = []
        sentence2 = []
        label = []
        for b in batch:
            sentence = " ".join(b[0])
            sentence += " ".join(b[1])
            sentence += " ".join(b[2])
            sentence += " ".join(b[3])
            # Concatenate the story for only one sentence
            ref_sentences.append(sentence)
            sentence1.append(" ".join(b[4]))
            sentence2.append(" ".join(b[5]))
            label.append(int(b[6][0]) - 1)
        ref_sentences, sentence1 = np.array(ref_sentences, dtype=object), np.array(sentence1, dtype=object)
        sentence2 = np.array(sentence1, dtype=object)
        with self.graph.as_default():
            # Get the elmo embeddings for the input sentences and ref sentences (stories)
            sent1_emb = self.elmo_model.predict(sentence1, batch_size=len(batch))
            sent2_emb = self.elmo_model.predict(sentence2, batch_size=len(batch))
            ref_sentences_emb = self.elmo_model.predict(ref_sentences, batch_size=len(batch))
            sent2_pred = self.type_translation_model.predict(
                    [ref_sentences_emb, sent1_emb],
                    batch_size=len(batch))
            sent1_pred = self.type_translation_model.predict(
                    [ref_sentences_emb, sent2_emb],
                    batch_size=len(batch))
        count_correct = 0
        for b in range(len(batch)):
            diff_sent2 = np.linalg.norm(sent2_pred - sent2_emb)
            diff_sent1 = np.linalg.norm(sent1_pred - sent1_emb)
            if diff_sent2 < diff_sent1:  # then 2 is the wrong one
                if not label[b]:
                    count_correct += 1
            else:
                if label[b]:
                    count_correct += 1

        return float(count_correct) / float(len(batch))


def get_elmo_embedding(elmo_fn):
    elmo_embeddings = keras.layers.Lambda(elmo_fn, output_shape=(1024,))
    sentence = keras.layers.Input(shape=(1,), dtype="string")
    sentence_emb = elmo_embeddings(sentence)
    model = keras.models.Model(inputs=sentence, outputs=sentence_emb)
    return model
