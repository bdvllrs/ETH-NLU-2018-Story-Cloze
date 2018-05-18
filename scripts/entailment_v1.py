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
import keras
from keras import backend as keras_backed
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from utils import SNLIDataloader
from nltk import word_tokenize


def preprocess_fn(line):
    # label = [entailment, neutral, contradiction]
    label = [1, 0, 0]
    if line['gold_label'] == 'contradiction':
        label = [0, 0, 1]
    elif line['gold_label'] == 'neutral':
        label = [0, 1, 0]
    sentence1 = ' '.join(word_tokenize(line['sentence1']))
    sentence2 = ' '.join(word_tokenize(line['sentence2']))
    output = [label, sentence1, sentence2]
    return output


def output_fn(batch):
    batch = np.array(batch, dtype=object)
    return [batch[:, 1], batch[:, 2]], np.array(list(batch[:, 0]))


class ElmoEmbedding:
    def __init__(self, elmo_model):
        self.elmo_model = elmo_model
        self.__name__ = "elmo_embeddings"

    def __call__(self, x):
        return self.elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)[
            "default"]


# class Elmo(keras.engine.topology.Layer):
#     def __init__(self, elmo_model, **kwargs):
#         self.elmo_model = elmo_model
#         super(Elmo, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         super(Elmo, self).build(input_shape)  # Be sure to call this at the end
#
#     def call(self, x):
#         return self.elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)[
#             "default"]
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], 1024)


def model(sess, config):
    if config.debug:
        print('Importing Elmo module...')
    elmo_model = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
    if config.debug:
        print('Imported.')
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    # TODO: Try with the `word_emb` to have all embeddings (size: max_length, 1024)
    elmo_embeddings = ElmoEmbedding(elmo_model)
    # elmo_embeddings = lambda x: elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)[
    #     "default"]

    # def elmo_embeddings(x):
    #     return elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

    dense_layer_1 = keras.layers.Dense(500, activation='relu')
    dense_layer_2 = keras.layers.Dense(100, activation='relu')
    dense_layer_3 = keras.layers.Dense(3, activation='sigmoid')

    sentence_1 = keras.layers.Input(shape=(1,), dtype=tf.string)  # Sentences comes in as a string
    sentence_2 = keras.layers.Input(shape=(1,), dtype=tf.string)
    embedding = keras.layers.Lambda(elmo_embeddings, output_shape=(1024,))
    sentence_1_embedded = embedding(sentence_1)
    sentence_2_embedded = embedding(sentence_2)

    # Graph
    inputs = keras.layers.Concatenate()([sentence_1_embedded, sentence_2_embedded])
    # inputs = sentiments
    output = keras.layers.Dropout(0.3)(dense_layer_1(inputs))
    output = keras.layers.Dropout(0.3)(dense_layer_2(output))
    output = dense_layer_3(output)

    # Model
    model = keras.models.Model(inputs=[sentence_1, sentence_2], outputs=[output])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    return model


def main(config):
    train_set = SNLIDataloader('data/snli_1.0/snli_1.0_train.jsonl')
    train_set.set_preprocess_fn(preprocess_fn)
    train_set.set_output_fn(output_fn)
    dev_set = SNLIDataloader('data/snli_1.0/snli_1.0_dev.jsonl')
    dev_set.set_preprocess_fn(preprocess_fn)
    dev_set.set_output_fn(output_fn)
    test_set = SNLIDataloader('data/snli_1.0/snli_1.0_test.jsonl')

    generator_training = train_set.get_batch(config.batch_size, config.n_epochs)
    generator_dev = dev_set.get_batch(config.batch_size, config.n_epochs)

    # Initialize tensorflow session
    sess = tf.Session()
    keras_backed.set_session(sess)  # Set to keras backend

    keras_model = model(sess, config)

    verbose = 0 if not config.debug else 1
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Callbacks
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/' + timestamp + '-entailmentv1/', histogram_freq=0,
                                              batch_size=config.batch_size,
                                              write_graph=False,
                                              write_grads=True)

    model_path = os.path.abspath(
        os.path.join(os.curdir, './builds/' + timestamp))
    model_path += '-entailmentv1_checkpoint_epoch-{epoch:02d}.hdf5'

    saver = keras.callbacks.ModelCheckpoint(model_path,
                                            monitor='val_acc', verbose=verbose, save_best_only=True)

    keras_model.fit_generator(generator_training, steps_per_epoch=2,
                              epochs=config.n_epochs,
                              verbose=verbose,
                              validation_data=generator_dev,
                              validation_steps=1,
                              callbacks=[tensorboard, saver])
