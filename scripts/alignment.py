import datetime
import os
import random

import keras
import tensorflow as tf
import tensorflow_hub as hub
import keras.backend as K
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LeakyReLU, BatchNormalization, Lambda

from utils import Dataloader, SNLIDataloaderPairs
from scripts import DefaultScript


class Script(DefaultScript):
    slug = 'alignment'

    def test(self):
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

        # If we gave the models to the encoder decodes...
        self.use_pretrained_models = self.config.alignment.is_set(
                'decoder_target_model') and self.config.alignment.is_set(
                'decoder_src_model') and self.config.alignment.is_set(
                'encoder_target_model') and self.config.alignment.is_set('encoder_src_model')

        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        self.graph = tf.get_default_graph()

        elmo_emb_fn = ElmoEmbedding(elmo_model)

        elmo_embeddings = keras.layers.Lambda(elmo_emb_fn, output_shape=(1024,))
        sentence = keras.layers.Input(shape=(1,), dtype="string")
        sentence_emb = elmo_embeddings(sentence)

        self.elmo_model = keras.models.Model(inputs=sentence, outputs=sentence_emb)

        test_set = Dataloader(self.config, 'data/test_stories.csv', testing_data=True)
        test_set.load_dataset('data/test.bin')
        test_set.load_vocab('./data/default.voc', self.config.vocab_size)
        test_set.set_output_fn(self.output_fn_test)

        generator_test = test_set.get_batch(self.config.batch_size, self.config.n_epochs)

        model = keras.models.load_model(self.config.alignment.final_model)

        print(model.metrics_names)
        acc_targets = []
        acc_srcs = []
        for inputs, labels in generator_test:
            results = model.evaluate(inputs, labels, batch_size=len(inputs))
            acc_target, acc_src = results[-4], results[-5]
            acc_targets.append(acc_target)
            acc_srcs.append(acc_src)
            print(np.mean(acc_targets), np.mean(acc_srcs))

    def train(self):
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

        # If we gave the models to the encoder decodes...
        self.use_pretrained_models = self.config.alignment.is_set(
                'decoder_target_model') and self.config.alignment.is_set(
                'decoder_src_model') and self.config.alignment.is_set(
                'encoder_target_model') and self.config.alignment.is_set('encoder_src_model')

        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        self.graph = tf.get_default_graph()

        elmo_emb_fn = ElmoEmbedding(elmo_model)

        elmo_embeddings = keras.layers.Lambda(elmo_emb_fn, output_shape=(1024,))
        sentence = keras.layers.Input(shape=(1,), dtype="string")
        sentence_emb = elmo_embeddings(sentence)

        self.elmo_model = keras.models.Model(inputs=sentence, outputs=sentence_emb)

        train_set = SNLIDataloaderPairs('data/snli_1.0/snli_1.0_train.jsonl')
        train_set.set_preprocess_fn(preprocess_fn)
        train_set.load_vocab('./data/snli_vocab.dat', self.config.vocab_size)
        train_set.set_output_fn(self.output_fn)

        test_set = Dataloader(self.config, 'data/test_stories.csv', testing_data=True)
        test_set.load_dataset('data/test.bin')
        test_set.load_vocab('./data/default.voc', self.config.vocab_size)
        test_set.set_output_fn(self.output_fn_test)

        generator_training = train_set.get_batch(self.config.batch_size, self.config.n_epochs)
        generator_dev = test_set.get_batch(self.config.batch_size, self.config.n_epochs)

        self.define_models()

        model = self.build_graph()
        frozen_model = self.build_frozen_graph()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer = tf.summary.FileWriter('./logs/' + timestamp + '-alignment/', self.graph)

        model_path = os.path.abspath(
                os.path.join(os.curdir, './builds/' + timestamp))
        model_path += '-alignment-model_checkpoint_step-'

        last_created_file = None

        self.use_frozen = False
        min_source_loss = None

        for k, (inputs, labels) in enumerate(generator_training):
            # We train the frozen model and the unfrozen model jointly
            if self.use_frozen:
                # Generator training
                metrics = frozen_model.train_on_batch(inputs, labels)
                if not k % self.config.print_train_every:
                    print_on_tensorboard(writer, frozen_model.metrics_names, metrics, k, 'train_f')
            else:
                metrics = model.train_on_batch(inputs, labels)
                if not k % self.config.print_train_every:
                    print_on_tensorboard(writer, model.metrics_names, metrics, k, 'train_uf')

            self.use_frozen = not self.use_frozen
            print(k, k % self.config.test_and_save_every, not k % self.config.test_and_save_every)

            if not k % self.config.test_and_save_every:
                test_metrics = []
                for j, (inputs_val, labels_val) in enumerate(generator_dev):
                    if j >= self.config.limit_test_step:
                        break
                    test_metrics.append(frozen_model.test_on_batch(inputs_val, labels_val))
                test_metrics = np.mean(test_metrics, axis=0)
                # Save value to tensorboard
                print_on_tensorboard(writer, frozen_model.metrics_names, test_metrics, k, 'test')
                test_metrics_dict = get_dict_from_lists(frozen_model.metrics_names, test_metrics)
                # We save the model is loss is better for generator
                # We only want to save the generator model
                if min_source_loss is None or test_metrics_dict['disrc_src_loss'] < min_source_loss:
                    frozen_model.save(model_path + str(k) + ".hdf5")
                    if last_created_file is not None:
                        os.remove(last_created_file)  # Only keep the best one
                    last_created_file = model_path + str(k) + ".hdf5"

    def define_models(self):
        # Decoder target
        input_target_decoder = Input((4096,))
        layer_1_target_decoder = Dense(4096)
        layer_2_target_decoder = Dense(2048, activation="relu")
        dec_target = EncoderDecoder(layer_1_target_decoder, layer_2_target_decoder)
        self.decoder_target_model = Model(input_target_decoder, dec_target(input_target_decoder))
        self.decoder_target_model.compile("adam", "binary_crossentropy")

        # Encoder src
        input_src_encoder = Input((2048,))
        layer_1_src_encoder = Dense(2048)
        layer_2_src_encoder = Dense(4096, activation="relu")
        encoder_src = EncoderDecoder(layer_1_src_encoder, layer_2_src_encoder)
        self.encoder_src_model = Model(input_src_encoder, encoder_src(input_src_encoder))
        self.encoder_src_model.compile("adam", "binary_crossentropy")

        # Decoder src
        input_src_decoder = Input((4096,))
        layer_1_src_decoder = Dense(4096)
        layer_2_src_decoder = Dense(2048, activation="relu")
        decoder_src = EncoderDecoder(layer_1_src_decoder, layer_2_src_decoder)
        self.decoder_src_model = Model(input_src_decoder, decoder_src(input_src_decoder))
        self.decoder_src_model.compile("adam", "binary_crossentropy")

        # Encoder target
        input_target_encoder = Input((2048,))
        layer_1_target_encoder = Dense(2048)
        layer_2_target_encoder = Dense(4096, activation="relu")
        encoder_target = EncoderDecoder(layer_1_target_encoder, layer_2_target_encoder)
        self.encoder_target_model = Model(input_target_encoder, encoder_target(input_target_encoder))
        self.encoder_target_model.compile("adam", "binary_crossentropy")

        # Discriminator
        input_discriminator = Input((4096,))
        layer_1_discriminator = Dense(1026, name="discr_layer_1")
        layer_2_discriminator = Dense(512, name="discr_layer2")
        layer_3_discriminator = Dense(1, activation="sigmoid", name="discr_layer3")
        discriminator = EncoderDecoder(layer_1_discriminator, layer_2_discriminator, layer_3_discriminator,
                                       name="discriminator")
        self.discriminator = Model(input_discriminator, discriminator(input_discriminator))
        self.discriminator.compile("adam", "binary_crossentropy")

    def build_graph(self):
        input_src_ori = Input((1024,))  # src sentence (only last sentence of story)
        input_src_noise_ori = Input((1024,))  # Noise on src sentence
        input_target_ori = Input((1024,))  # Noise on target sentence
        input_target_noise_ori = Input((1024,))  # Noise on target sentence
        history_ref_ori = Input((1024,))  # Target of the story

        input_src = keras.layers.concatenate([input_src_ori, history_ref_ori])
        input_src_noise = keras.layers.concatenate([input_src_noise_ori, history_ref_ori])
        input_target_noise = keras.layers.concatenate([input_target_noise_ori, history_ref_ori])
        input_target = keras.layers.concatenate([input_target_ori, history_ref_ori])

        # Build graph
        src = self.encoder_src_model(input_src_noise)
        out_src = self.decoder_src_model(src)  # Must be equal to input_src

        target = self.encoder_target_model(input_target_noise)
        out_target = self.decoder_target_model(target)  # Must be equal to input_target

        discriminator_src = Lambda(lambda x: x, name="disrc_src")(
                self.discriminator(src))  # 0 src annd from src_enc or target and from target_enc, 1 otherwise
        discriminator_target = Lambda(lambda x: x, name="disrc_target")(self.discriminator(target))

        # Calculate differences
        diff_out_src_input_src = keras.layers.subtract([out_src, input_src])
        diff_out_target_input_target = keras.layers.subtract([out_target, input_target])

        dist_src = keras.layers.dot([diff_out_src_input_src, diff_out_src_input_src], axes=1, name="dist_src")
        dist_target = keras.layers.dot([diff_out_target_input_target, diff_out_target_input_target], axes=1,
                                       name="dist_target")

        # Get pretrained non trainable encoders & decoders
        encoder_src_ntrainable = self.encoder_src(self.encoder_src_model)
        encoder_target_ntrainable = self.encoder_target(self.encoder_target_model)
        decoder_target_ntrainable = self.decoder_target(self.decoder_target_model)
        decoder_src_ntrainable = self.decoder_src(self.decoder_src_model)

        out_nn_target = decoder_target_ntrainable(encoder_src_ntrainable(input_src))  # Without noise
        out_nn_src = decoder_src_ntrainable(encoder_target_ntrainable(input_target))

        out_mix_target_enc = self.encoder_src_model(out_nn_src)
        out_mix_source_enc = self.encoder_target_model(out_nn_target)

        out_mix_target = self.decoder_target_model(out_mix_target_enc)
        out_mix_src = self.decoder_src_model(out_mix_source_enc)  # Must be equal to input_src

        discriminator_target_mix = self.discriminator(out_mix_target_enc)  # Needs to be 0
        discriminator_src_mix = self.discriminator(out_mix_source_enc)

        diff_out_mix_src_input_src = keras.layers.subtract([input_src, out_mix_src])
        diff_out_mix_target_input_target = keras.layers.subtract([input_target, out_mix_target])

        dist_mix_input_src = keras.layers.dot([diff_out_mix_src_input_src, diff_out_mix_src_input_src], axes=1,
                                              name="dist_mix_input_src")
        dist_mix_target_input_src = keras.layers.dot(
                [diff_out_mix_target_input_target, diff_out_mix_target_input_target], axes=1,
                name="dist_mix_target_input_src")

        model = Model(inputs=[input_src_ori, input_src_noise_ori, input_target_ori, input_target_noise_ori, history_ref_ori],
                      outputs=[dist_mix_input_src, dist_mix_target_input_src, dist_src, dist_target,
                               discriminator_src, discriminator_target, discriminator_target_mix,
                               discriminator_src_mix])
        model.compile("adam", "binary_crossentropy", ['accuracy'])
        return model

    def build_frozen_graph(self):
        input_src_ori = Input((1024,))  # src sentence (only last sentence of story)
        input_src_noise_ori = Input((1024,))  # Noise on src sentence
        input_target_ori = Input((1024,))  # Noise on target sentence
        input_target_noise_ori = Input((1024,))  # Noise on target sentence
        history_ref_ori = Input((1024,))  # Target of the story

        input_src = keras.layers.concatenate([input_src_ori, history_ref_ori])
        input_src_noise = keras.layers.concatenate([input_src_noise_ori, history_ref_ori])
        input_target_noise = keras.layers.concatenate([input_target_noise_ori, history_ref_ori])
        input_target = keras.layers.concatenate([input_target_ori, history_ref_ori])


        self.encoder_src_model.trainable = False
        self.encoder_target_model.trainable = False
        self.decoder_src_model.trainable = False
        self.decoder_target_model.trainable = False

        # Build graph
        src = self.encoder_src_model(input_src_noise)

        target = self.encoder_target_model(input_target_noise)

        discriminator_src = Lambda(lambda x: x, name="disrc_src")(
                self.discriminator(src))  # 0 src = true sentence, 1 target = wrong
        discriminator_target = Lambda(lambda x: x, name="disrc_target")(self.discriminator(target))

        # Get pretrained non trainable encoders & decoders
        encoder_src_ntrainable = self.encoder_src(self.encoder_src_model)
        encoder_target_ntrainable = self.encoder_target(self.encoder_target_model)
        decoder_target_ntrainable = self.decoder_target(self.decoder_target_model)
        decoder_src_ntrainable = self.decoder_src(self.decoder_src_model)

        out_nn_target = decoder_target_ntrainable(encoder_src_ntrainable(input_src))  # Without noise
        out_nn_src = decoder_src_ntrainable(encoder_target_ntrainable(input_target))

        out_mix_target_enc = self.encoder_src_model(out_nn_src)
        out_mix_source_enc = self.encoder_target_model(out_nn_target)

        discriminator_target_mix = self.discriminator(out_mix_target_enc)  # Needs to be 0
        discriminator_src_mix = self.discriminator(out_mix_source_enc)

        model = Model(inputs=[input_src_ori, input_src_noise_ori, input_target_ori, input_target_noise_ori, history_ref_ori],
                      outputs=[discriminator_src, discriminator_target, discriminator_target_mix,
                               discriminator_src_mix])
        model.compile("adam", "binary_crossentropy", ['accuracy'])
        return model

    def decoder_target(self, decoder):
        inp = Input((4096,))
        decoder.trainable = False
        out = decoder(inp)
        model = Model(inp, out)
        model.compile("adam", "binary_crossentropy")
        decoder.trainable = True
        return model

    def decoder_src(self, decoder):
        inp = Input((4096,))
        decoder.trainable = False
        out = decoder(inp)
        model = Model(inp, out)
        model.compile("adam", "binary_crossentropy")
        decoder.trainable = True
        return model

    def encoder_src(self, encoder):
        inp = Input((2048,))
        encoder.trainable = False
        out = encoder(inp)
        model = Model(inp, out)
        model.compile("adam", "binary_crossentropy")
        encoder.trainable = True
        return model

    def encoder_target(self, encoder):
        inp = Input((2048,))
        encoder.trainable = False
        out = encoder(inp)
        model = Model(inp, out)
        model.compile("adam", "binary_crossentropy")
        encoder.trainable = True
        return model

    def add_noise(self, variable, drop_probability: float = 0.1, shuffle_max_distance: int = 3):
        """
        :param variable:np array that : [[sentence1][sentence2]]
        :param drop_probability: we drop every word in the input sentence with a probability
        :param shuffle_max_distance: we slightly shuffle the input sentence
        :return:
        """
        variable = np.array([[variable]])

        def perm(i):
            return i[0] + (shuffle_max_distance + 1) * np.random.random()

        liste = []
        for b in range(variable.shape[0]):
            sequence = variable[b]
            if (type(sequence) != list):
                sequence = sequence.tolist()
            sequence, reminder = sequence[:-1], sequence[-1:]
            if len(sequence) != 0:
                compteur = 0
                for num, val in enumerate(np.random.random_sample(len(sequence))):
                    if val < drop_probability:
                        sequence.pop(num - compteur)
                        compteur = compteur + 1
                sequence = [x for _, x in sorted(enumerate(sequence), key=perm)]
            sequence = np.concatenate((sequence, reminder), axis=0)
            liste.append(sequence)
        new_variable = np.array(liste)
        return new_variable[0, 0]

    def embedding(self, x):
        with self.graph.as_default():
            result = self.elmo_model.predict(x, batch_size=len(x))
        return result

    def output_fn(self, _, batch):
        all_histoire_debut_embedding = []
        all_histoire_fin_embedding = []
        all_histoire_noise_debut = []
        all_histoire_noise_fin = []
        all_history_ref = []
        for b in batch:
            history_ref = b[0][0]
            histoire_debut = b[0][1]
            histoire_noise_debut = self.add_noise(histoire_debut)
            histoire_fin = b[1][1]
            histoire_noise_fin = self.add_noise(histoire_fin)
            all_history_ref.append(history_ref)
            if not self.use_frozen:  # We send the real values
                all_histoire_fin_embedding.append(histoire_fin)
                all_histoire_noise_debut.append(histoire_noise_debut)
                all_histoire_noise_fin.append(histoire_noise_fin)
                all_histoire_debut_embedding.append(histoire_debut)
            else:  # We mix them up
                all_histoire_fin_embedding.append(histoire_debut)
                all_histoire_noise_debut.append(histoire_noise_fin)
                all_histoire_noise_fin.append(histoire_noise_debut)
                all_histoire_debut_embedding.append(histoire_fin)
        all_histoire_fin_embedding = self.embedding(np.array(all_histoire_fin_embedding))
        all_histoire_debut_embedding = self.embedding(np.array(all_histoire_debut_embedding))
        all_histoire_noise_fin = self.embedding(np.array(all_histoire_noise_fin))
        all_histoire_noise_debut = self.embedding(np.array(all_histoire_noise_debut))
        all_history_ref_embedding = self.embedding(np.array(all_history_ref))
        if self.use_frozen:  # We swithed up the right and bad ones
            ones = np.ones(len(batch))
            # disriminator bust be at one because inverted
            return [all_histoire_debut_embedding, all_histoire_noise_debut,
                    all_histoire_fin_embedding, all_histoire_noise_fin, all_history_ref_embedding], [ones, ones, ones,
                                                                                                     ones]
        zeros = np.zeros(len(batch))
        return [all_histoire_debut_embedding, all_histoire_noise_debut,
                all_histoire_fin_embedding, all_histoire_noise_fin, all_history_ref_embedding], [zeros, zeros, zeros,
                                                                                                 zeros, zeros, zeros,
                                                                                                 zeros,
                                                                                                 zeros]

    def output_fn_test(self, data):
        """
        :param data:
        :return:
        """
        batch = np.array(data.batch)
        all_histoire_debut_embedding = []
        all_histoire_fin_embedding = []
        all_histoire_noise_debut = []
        all_histoire_noise_fin = []
        all_history_ref = []
        label1 = []
        label2 = []
        for b in batch:
            all_history_ref.append(" ".join(b[3]))
            histoire_debut = " ".join(b[4])
            histoire_noise_debut = self.add_noise(histoire_debut)
            histoire_fin = " ".join(b[5])
            histoire_noise_fin = self.add_noise(histoire_fin)
            all_histoire_debut_embedding.append(histoire_debut)
            all_histoire_fin_embedding.append(histoire_fin)
            all_histoire_noise_debut.append(histoire_noise_debut)
            all_histoire_noise_fin.append(histoire_noise_fin)
            label = int(b[6][0]) - 1
            # 0 if beginning = src = true sentence
            # 1 if beginning = target = false sentence
            label1.append(label)
            label2.append(1 - label)
        all_histoire_fin_embedding = self.embedding(np.array(all_histoire_fin_embedding))
        all_histoire_debut_embedding = self.embedding(np.array(all_histoire_debut_embedding))
        all_histoire_noise_fin = self.embedding(np.array(all_histoire_noise_fin))
        all_histoire_noise_debut = self.embedding(np.array(all_histoire_noise_debut))
        all_history_ref_embedding = self.embedding(np.array(all_history_ref))
        label1 = np.array(label1)
        label2 = np.array(label2)
        return [all_histoire_debut_embedding,
                all_histoire_noise_debut,
                all_histoire_fin_embedding,
                all_histoire_noise_fin, all_history_ref_embedding], [label1, label2, label1, label2]


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


class EncoderDecoder:
    def __init__(self, layer1, layer2, layer3=None, name=None):
        self.layer2 = layer2
        self.layer1 = layer1
        self.layer3 = layer3
        if name is not None and layer3 is not None:
            self.layer3.name = name
        elif name is not None:
            self.layer2.name = name

    def __call__(self, x):
        l1 = BatchNormalization()(Dropout(0.3)(LeakyReLU()(self.layer1(x))))
        if self.layer3 is not None:
            l2 = BatchNormalization()(Dropout(0.3)(LeakyReLU()(self.layer2(l1))))
            return BatchNormalization()(self.layer3(l2))
        else:
            return BatchNormalization()(self.layer2(l1))


def print_on_tensorboard(writer, metrics, results, k, prefix=""):
    """
    Add values to summary
    :param writer: tensroflow writer
    :param metrics: metric names
    :param results: metric values
    :param k: x axis
    :param prefix: prefix to add the names
    """
    print("Saving on tensorboard...")
    # Save value to tensorboard
    accuracy_summary = tf.Summary()
    for name, value in zip(metrics, results):
        accuracy_summary.value.add(tag=prefix + "_" + name, simple_value=value)
    writer.add_summary(accuracy_summary, k)
    print("Saved.")


def get_dict_from_lists(keys, values):
    """
    Construct a dict from two lists
    :param keys:
    :param values:
    :return:
    """
    result = {}
    for name, value in zip(keys, values):
        result[name] = value
    return result
