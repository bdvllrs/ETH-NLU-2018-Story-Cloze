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
    slug = 'type_translation_gan'

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
    dense_layer_1 = keras.layers.Dense(4000, activation='relu')
    dense_layer_2 = keras.layers.Dense(3000, activation='relu')
    dense_layer_3 = keras.layers.Dense(1024, activation='tanh')

    sentence_ref = keras.layers.Input(shape=(1024,))
    sentence_neutral = keras.layers.Input(shape=(1024,))

    sentence = keras.layers.concatenate([sentence_ref, sentence_neutral])

    # inputs = sentiments
    output = keras.layers.Dropout(0.3)(dense_layer_1(sentence))
    output = keras.layers.Dropout(0.3)(dense_layer_2(output))
    output = dense_layer_3(output)

    # Model
    model = keras.models.Model(inputs=[sentence_ref, sentence_neutral], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    return model


def discriminator_model():
    dense_layer_1 = keras.layers.Dense(1000, activation='relu')
    dense_layer_2 = keras.layers.Dense(500, activation='relu')
    dense_layer_3 = keras.layers.Dense(1, activation='sigmoid')
    sentence_ref = keras.layers.Input(shape=(1024,))
    sentence_out = keras.layers.Input(shape=(1024,))

    sentence = keras.layers.concatenate([sentence_ref, sentence_out])

    output = keras.layers.Dropout(0.3)(dense_layer_1(sentence))
    output = keras.layers.Dropout(0.3)(dense_layer_2(output))
    output = dense_layer_3(output)

    # Model
    model = keras.models.Model(inputs=[sentence_ref, sentence_out], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    return model


def combined_model(g_model, d_model):
    sentence_ref = keras.layers.Input(shape=(1024,))
    sentence_neutral = keras.layers.Input(shape=(1024,))

    sentence_out = g_model([sentence_ref, sentence_neutral])
    d_model.trainable = False  # Do not train d_model during this step
    valid = d_model([sentence_ref, sentence_out])

    c_model = keras.models.Model([sentence_ref, sentence_neutral], valid)
    c_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    return c_model


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

    # Models

    g_model = generator_model()  # Generator model
    d_model = discriminator_model()  # Discriminator model
    c_model = combined_model(g_model, d_model)  # Combined model

    # Training

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    writer = tf.summary.FileWriter('./logs/' + timestamp + '-type-translation-gan/', graph)

    model_path = os.path.abspath(
            os.path.join(os.curdir, './builds/' + timestamp))
    model_path += '-type-translation-gan-g-model_checkpoint_step-'

    last_created_file = None

    d_loss, g_loss = None, None
    d_acc, g_acc = None, None
    min_g_loss = None

    for k, ((ref_sent, neutral_sent), real_neg_sentence) in enumerate(generator_training):
        # We train the discriminator and generator one time step after the other
        if k % 2:
            # Discriminator training
            fake_neg_sentence = g_model.predict([ref_sent, neutral_sent], batch_size=config.batch_size)
            d_loss_real, d_acc_real = d_model.train_on_batch([ref_sent, real_neg_sentence],
                                                             np.ones(config.batch_size))  # Real negative endings
            d_loss_fakes, d_acc_fakes = d_model.train_on_batch([ref_sent, fake_neg_sentence],
                                                               np.zeros(
                                                                       config.batch_size))  # Generated negative endings
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fakes)
            d_acc = 0.5 * np.add(d_acc_real, d_acc_fakes)
        else:
            # Generator training
            g_loss, g_acc = c_model.train_on_batch([ref_sent, neutral_sent],
                                                   np.ones(config.batch_size))  # We want the d_model to fail

        if k > 0 and not k % config.test_and_save_every:
            # Testing and saving to tensorboard.
            g_loss_dev, d_loss_dev = [], []
            g_acc_dev, d_acc_dev = [], []
            for j, ((ref_sent_test, neutral_sent_test), real_neg_sent_test) in enumerate(generator_dev):
                if j == 5:
                    break
                # Discriminator training
                fake_neg_sentence_test = g_model.predict([ref_sent_test, neutral_sent_test],
                                                         batch_size=config.batch_size)
                d_loss_real, d_acc_real = d_model.test_on_batch([ref_sent_test, real_neg_sent_test],
                                                                np.ones(config.batch_size))  # Real negative endings
                d_loss_fakes, d_acc_fakes = d_model.test_on_batch([ref_sent_test, fake_neg_sentence_test],
                                                                  np.zeros(
                                                                          config.batch_size))  # Generated negative endings
                d_loss_dev.append(0.5 * np.add(d_loss_real, d_loss_fakes))
                d_acc_dev.append(0.5 * np.add(d_acc_real, d_acc_fakes))
                # Generator training
                g_loss_dev_one, g_acc_dev_one = c_model.test_on_batch([ref_sent, neutral_sent],
                                                              np.ones(config.batch_size)
                                                              )  # We want the d_model to fail

                g_loss_dev.append(g_loss_dev_one)
                g_acc_dev.append(g_acc_dev_one)
            g_mean_loss, d_mean_loss = np.mean(g_loss_dev), np.mean(d_loss_dev)
            g_mean_acc, d_mean_acc = np.mean(g_acc_dev), np.mean(d_acc_dev)

            # Save value to tensorboard
            accuracy_summary = tf.Summary()
            accuracy_summary.value.add(tag='train_loss_discriminator', simple_value=d_loss)
            accuracy_summary.value.add(tag='train_loss_generator', simple_value=g_loss)
            accuracy_summary.value.add(tag='train_acc_generator', simple_value=d_acc)
            accuracy_summary.value.add(tag='train_acc_discriminator', simple_value=d_acc)
            accuracy_summary.value.add(tag='test_loss_discriminator', simple_value=d_mean_loss)
            accuracy_summary.value.add(tag='test_loss_generator', simple_value=g_mean_loss)
            accuracy_summary.value.add(tag='test_acc_generator', simple_value=g_mean_acc)
            accuracy_summary.value.add(tag='test_acc_discriminator', simple_value=d_mean_loss)
            writer.add_summary(accuracy_summary, k)

            # We save the model is loss is better for generator
            # We only want to save the generator model
            if min_g_loss is None or g_mean_loss < min_g_loss:
                g_model.save(model_path + str(k) + ".hdf5")
                if last_created_file is not None:
                    os.remove(last_created_file)  # Only keep the best one
                last_created_file = model_path + str(k) + ".hdf5"
