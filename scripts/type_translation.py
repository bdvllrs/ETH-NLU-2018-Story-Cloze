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
from utils import SNLIDataloaderPairs
from scripts import DefaultScript


class Script(DefaultScript):
    slug = 'type_translation'

    def train(self):
        main(self.config)


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
        output_sentences = np.array(output_sentences, dtype=object)
        with self.graph.as_default():
            out_sent = self.elmo_emb_model.predict(output_sentences, batch_size=8)
        return [np.array(ref_sentences, dtype=object), np.array(input_sentences, dtype=object)], out_sent


def get_elmo_embedding(elmo_fn):
    elmo_embeddings = keras.layers.Lambda(elmo_fn, output_shape=(1024,))
    sentence = keras.layers.Input(shape=(1,), dtype="string")
    sentence_emb = elmo_embeddings(sentence)
    model = keras.models.Model(inputs=sentence, outputs=sentence_emb)
    return model


def model(elmo_fn):
    elmo_embeddings = keras.layers.Lambda(elmo_fn, output_shape=(1024,))

    dense_layer_1 = keras.layers.Dense(4000, activation='relu')
    dense_layer_2 = keras.layers.Dense(3000, activation='relu')
    dense_layer_3 = keras.layers.Dense(1024, activation='relu')

    sentence_ref = keras.layers.Input(shape=(1,), dtype="string")
    sentence_neutral = keras.layers.Input(shape=(1,), dtype="string")
    sentence_ref_emb = elmo_embeddings(sentence_ref)
    sentence_neutral_emb = elmo_embeddings(sentence_neutral)

    sentence = keras.layers.concatenate([sentence_ref_emb, sentence_neutral_emb])

    # inputs = sentiments
    output = keras.layers.Dropout(0.3)(dense_layer_1(sentence))
    output = keras.layers.Dropout(0.3)(dense_layer_2(output))
    output = dense_layer_3(output)

    # Model
    model = keras.models.Model(inputs=[sentence_ref, sentence_neutral], outputs=output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
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

    keras_model = model(elmo_emb_fn)

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

    keras_model.fit_generator(generator_training, steps_per_epoch=100,
                              epochs=config.n_epochs,
                              verbose=verbose,
                              validation_data=generator_dev,
                              validation_steps=len(dev_set)/config.batch_size,
                              callbacks=[tensorboard, saver])
