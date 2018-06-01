"""
Credits to
- Matteo Pagliardini, Prakhar Gupta, Martin Jaggi,
    Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features NAACL 2018,
    https://arxiv.org/abs/1703.02507
- A large annotated corpus for learning natural language inference,
    Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning,
    https://nlp.stanford.edu/pubs/snli_paper.pdf.
"""
import datetime
import os
import keras
import tensorflow as tf
import tensorflow_hub as hub
import keras.backend as K
import numpy as np
from keras.layers import BatchNormalization, Dropout, LeakyReLU

from utils import SNLIDataloaderPairs
from scripts import DefaultScript


class Script(DefaultScript):
    slug = 'type_translation'

    def train(self):
        main(self.config)

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

        output_fn = OutputFN(elmo_model_emb, graph)

        test_set = SNLIDataloaderPairs('data/snli_1.0/snli_1.0_test.jsonl')
        test_set.set_preprocess_fn(preprocess_fn)
        test_set.set_output_fn(output_fn)

        generator_test = test_set.get_batch(self.config.batch_size, self.config.n_epochs)

        # keras_model = model(elmo_emb_fn)

        verbose = 0 if not self.config.debug else 1

        keras_model = keras.models.load_model(
                './builds/leonhard/2018-05-30 15:22:53-type-translation_checkpoint_epoch-77.hdf5', {
                    'elmo_embeddings': elmo_emb_fn
                })
        loss = keras_model.evaluate_generator(generator_test, steps=len(test_set) / self.config.batch_size,
                                              verbose=verbose)
        print(loss)


class ElmoEmbedding:
    def __init__(self, elmo_model):
        self.elmo_model = elmo_model
        self.__name__ = "elmo_embeddings"

    def __call__(self, x):
        return self.elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)[
            "default"]


def preprocess_fn(line):
    output = [line['sentence1'], line['sentence2']]
    return output


class OutputFN:
    def __init__(self, elmo_emb_model, graph):
        self.graph = graph
        self.elmo_emb_model = elmo_emb_model

    def __call__(self, _, batch):
        ref_sentences = []
        input_sentences = []
        output_sentences = []
        for b in batch:
            ref_sentences.append(b[0][0])
            input_sentences.append(b[0][1])
            output_sentences.append(b[1][1])
        ref_sentences = np.array(ref_sentences, dtype=object)
        input_sentences = np.array(input_sentences, dtype=object)
        output_sentences = np.array(output_sentences, dtype=object)
        with self.graph.as_default():
            ref_sent = self.elmo_emb_model.predict(ref_sentences, batch_size=len(batch))
            input_sent = self.elmo_emb_model.predict(input_sentences, batch_size=len(batch))
            out_sent = self.elmo_emb_model.predict(output_sentences, batch_size=len(batch))
        return [ref_sent, input_sent], out_sent


def get_elmo_embedding(elmo_fn):
    elmo_embeddings = keras.layers.Lambda(elmo_fn, output_shape=(1024,))
    sentence = keras.layers.Input(shape=(1,), dtype="string")
    sentence_emb = elmo_embeddings(sentence)
    model = keras.models.Model(inputs=sentence, outputs=sentence_emb)
    return model


def generator_model():
    dense_layer_1 = keras.layers.Dense(4096)
    dense_layer_2 = keras.layers.Dense(2048)
    dense_layer_3 = keras.layers.Dense(1024, activation='tanh')

    sentence_ref = keras.layers.Input(shape=(1024,))
    sentence_neutral = keras.layers.Input(shape=(1024,))

    sentence = keras.layers.concatenate([sentence_ref, sentence_neutral])

    # inputs = sentiments
    output = BatchNormalization(momentum=0.8)(Dropout(0.4)(LeakyReLU(alpha=0.2)(dense_layer_1(sentence))))
    output = BatchNormalization(momentum=0.8)(Dropout(0.4)(LeakyReLU(alpha=0.2)(dense_layer_2(output))))
    output = dense_layer_3(output)

    # Model
    model = keras.models.Model(inputs=[sentence_ref, sentence_neutral], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0002, decay=8e-9), loss="mean_squared_error",
                  metrics=['accuracy'])
    return model


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

    output_fn = OutputFN(elmo_model_emb, graph)

    train_set = SNLIDataloaderPairs('data/snli_1.0/snli_1.0_train.jsonl')
    train_set.load_vocab('./data/snli_vocab.dat', config.vocab_size)
    train_set.set_preprocess_fn(preprocess_fn)
    train_set.set_output_fn(output_fn)
    dev_set = SNLIDataloaderPairs('data/snli_1.0/snli_1.0_dev.jsonl')
    dev_set.load_vocab('./data/snli_vocab.dat', config.vocab_size)
    dev_set.set_preprocess_fn(preprocess_fn)
    dev_set.set_output_fn(output_fn)
    # test_set = SNLIDataloader('data/snli_1.0/snli_1.0_test.jsonl')

    generator_training = train_set.get_batch(config.batch_size, config.n_epochs)
    generator_dev = dev_set.get_batch(config.batch_size, config.n_epochs)

    keras_model = generator_model()

    verbose = 0 if not config.debug else 1
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Callbacks
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/' + timestamp + '-type-translation/', histogram_freq=0,
                                              batch_size=config.batch_size,
                                              write_graph=False,
                                              write_grads=True)

    model_path = os.path.abspath(
            os.path.join(os.curdir, './builds/' + timestamp))
    model_path += '-type-translation_checkpoint_epoch-{epoch:02d}.hdf5'

    saver = keras.callbacks.ModelCheckpoint(model_path,
                                            monitor='val_loss', verbose=verbose, save_best_only=True)

    keras_model.fit_generator(generator_training, steps_per_epoch=len(train_set)/config.batch_size,
                              epochs=config.n_epochs,
                              verbose=verbose,
                              validation_data=generator_dev,
                              validation_steps=len(dev_set)/config.batch_size,
                              callbacks=[tensorboard, saver])
