import datetime
import os

import keras
import tensorflow as tf
import tensorflow_hub as hub
import keras.backend as K
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LeakyReLU, BatchNormalization

from utils import Dataloader, SNLIDataloaderPairs
from scripts import DefaultScript


class Script(DefaultScript):
    slug = 'alignment'

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
        generator_test = test_set.get_batch(self.config.batch_size, self.config.n_epochs)

        model = self.build_graph()

        verbose = 0 if not self.config.debug else 1
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Callbacks
        tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/' + timestamp + '-alignment/', histogram_freq=0,
                                                  batch_size=self.config.batch_size,
                                                  write_graph=False,
                                                  write_grads=True)

        model_path = os.path.abspath(
                os.path.join(os.curdir, './builds/' + timestamp))
        model_path += '-alignment_checkpoint_epoch-{epoch:02d}.hdf5'

        saver = keras.callbacks.ModelCheckpoint(model_path,
                                                monitor='val_loss', verbose=verbose, save_best_only=True)

        model.fit_generator(generator_training, steps_per_epoch=len(train_set) / self.config.batch_size,
                            epochs=self.config.n_epochs,
                            verbose=verbose,
                            validation_data=generator_test,
                            validation_steps=len(test_set) / self.config.batch_size,
                            callbacks=[tensorboard, saver])

        if self.config.alignment.is_set('save_encoder_decoder') and self.config.alignment.save_encoder_decoder:
            print('saving models')
            model_path = os.path.abspath(
                    os.path.join(os.curdir, './builds/' + timestamp))
            model_path += '-alignment_'

            self.decoder_target_model.save(model_path + 'decoder_target.hdf5')
            self.decoder_src_model.save(model_path + 'decoder_src.hdf5')
            self.encoder_target_model.save(model_path + 'encoder_target.hdf5')
            self.encoder_src_model.save(model_path + 'encoder_src.hdf5')

    def build_graph(self):
        input_src = Input((1024,))  # src sentence (only last sentence of story)
        input_src_noise = Input((1024,))  # Noise on src sentence
        input_target = Input((1024,))  # Noise on target sentence
        input_target_noise = Input((1024,))  # Noise on target sentence

        if self.use_pretrained_models:
            self.decoder_target_model = keras.models.load_model(self.config.alignment.decoder_target_model)
            self.encoder_target_model = keras.models.load_model(self.config.alignment.encoder_target_model)
            self.decoder_src_model = keras.models.load_model(self.config.alignment.decoder_src_model)
            self.encoder_src_model = keras.models.load_model(self.config.alignment.encoder_src_model)
        else:
            # Decoder target
            input_target_decoder = Input((2048,))
            layer_1_target_decoder = Dense(2048)
            layer_2_target_decoder = Dense(1024)
            layer_3_target_decoder = Dense(1024, activation="relu")
            dec_target = EncoderDecoder(layer_1_target_decoder, layer_2_target_decoder, layer_3_target_decoder)
            self.decoder_target_model = Model(input_target_decoder, dec_target(input_target_decoder))
            self.decoder_target_model.compile("adam", "binary_crossentropy")

            # Encoder src
            input_src_encoder = Input((1024,))
            layer_1_src_encoder = Dense(1024)
            layer_2_src_encoder = Dense(1024)
            layer_3_src_encoder = Dense(2048, activation="relu")
            encoder_src = EncoderDecoder(layer_1_src_encoder, layer_2_src_encoder, layer_3_src_encoder)
            self.encoder_src_model = Model(input_src_encoder, encoder_src(input_src_encoder))
            self.encoder_src_model.compile("adam", "binary_crossentropy")

            # Decoder src
            input_src_decoder = Input((2048,))
            layer_1_src_decoder = Dense(2048)
            layer_2_src_decoder = Dense(2048)
            layer_3_src_decoder = Dense(1024, activation="relu")
            decoder_src = EncoderDecoder(layer_1_src_decoder, layer_2_src_decoder, layer_3_src_decoder)
            self.decoder_src_model = Model(input_src_decoder, decoder_src(input_src_decoder))
            self.decoder_src_model.compile("adam", "binary_crossentropy")

            # Encoder target
            input_target_encoder = Input((1024,))
            layer_1_target_encoder = Dense(1024)
            layer_2_target_encoder = Dense(1024)
            layer_3_target_encoder = Dense(2048, activation="relu")
            encoder_target = EncoderDecoder(layer_1_target_encoder, layer_2_target_encoder, layer_3_target_encoder)
            self.encoder_target_model = Model(input_target_encoder, encoder_target(input_target_encoder))
            self.encoder_target_model.compile("adam", "binary_crossentropy")

        # Discriminator
        layer_1_discriminator = Dense(1024)
        layer_2_discriminator = Dense(512)
        layer_3_discriminator = Dense(1, activation="sigmoid")
        discriminator = EncoderDecoder(layer_1_discriminator, layer_2_discriminator, layer_3_discriminator,
                                       name="discriminator")

        # Build graph
        src = self.encoder_src_model(input_src_noise)
        out_src = self.decoder_src_model(src)  # Must be equal to input_src

        target = self.encoder_target_model(input_target_noise)
        out_target = self.decoder_target_model(target)  # Must be equal to input_target

        discriminator_src = discriminator(src)  # 0 src = true sentence, 1 target = wrong
        discriminator_target = discriminator(target)

        # Calculate differences
        diff_out_src_input_src = keras.layers.subtract([out_src, input_src])
        diff_out_target_input_target = keras.layers.subtract([out_target, input_target])

        dist_src = keras.layers.dot([diff_out_src_input_src, diff_out_src_input_src], axes=1, name="dist_src")
        dist_target = keras.layers.dot([diff_out_target_input_target, diff_out_target_input_target], axes=1,
                                       name="dist_target")

        if self.use_pretrained_models:
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

            discriminator_target_mix = discriminator(out_mix_target_enc)  # Needs to be 0
            discriminator_src_mix = discriminator(out_mix_source_enc)

            diff_out_mix_src_input_src = keras.layers.subtract([input_src, out_mix_src])
            diff_out_mix_target_input_target = keras.layers.subtract([input_target, out_mix_target])

            dist_mix_input_src = keras.layers.dot([diff_out_mix_src_input_src, diff_out_mix_src_input_src], axes=1,
                                                  name="dist_mix_input_src")
            dist_mix_target_input_src = keras.layers.dot(
                    [diff_out_mix_target_input_target, diff_out_mix_target_input_target], axes=1,
                    name="dist_mix_target_input_src")

            model = Model(inputs=[input_src, input_src_noise, input_target, input_target_noise],
                          outputs=[dist_mix_input_src, dist_mix_target_input_src, dist_src, dist_target,
                                   discriminator_src, discriminator_target, discriminator_target_mix,
                                   discriminator_src_mix])
        else:
            model = Model(inputs=[input_src, input_src_noise, input_target, input_target_noise],
                          outputs=[dist_src, dist_target, discriminator_src, discriminator_target])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
        return model

    def decoder_target(self, decoder):
        inp = Input((2048,))
        decoder.trainable = False
        out = decoder(inp)
        model = Model(inp, out)
        model.compile("adam", "binary_crossentropy")
        decoder.trainable = True
        return model

    def decoder_src(self, decoder):
        inp = Input((2048,))
        decoder.trainable = False
        out = decoder(inp)
        model = Model(inp, out)
        model.compile("adam", "binary_crossentropy")
        decoder.trainable = True
        return model

    def encoder_src(self, encoder):
        inp = Input((1024,))
        encoder.trainable = False
        out = encoder(inp)
        model = Model(inp, out)
        model.compile("adam", "binary_crossentropy")
        encoder.trainable = True
        return model

    def encoder_target(self, encoder):
        inp = Input((1024,))
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
        for b in batch:
            histoire_debut = b[0][0] + " " + b[0][1]
            histoire_noise_debut = self.add_noise(histoire_debut)
            histoire_fin = b[1][0] + " " + b[1][1]
            histoire_noise_fin = self.add_noise(histoire_fin)
            all_histoire_fin_embedding.append(histoire_fin)
            all_histoire_noise_debut.append(histoire_noise_debut)
            all_histoire_noise_fin.append(histoire_noise_fin)
            all_histoire_debut_embedding.append(histoire_debut)
        all_histoire_fin_embedding = self.embedding(np.array(all_histoire_fin_embedding))
        all_histoire_debut_embedding = self.embedding(np.array(all_histoire_debut_embedding))
        all_histoire_noise_fin = self.embedding(np.array(all_histoire_noise_fin))
        all_histoire_noise_debut = self.embedding(np.array(all_histoire_noise_debut))
        zeros = np.zeros(len(batch))
        ones = np.ones(len(batch))
        if self.use_pretrained_models:
            return [np.array(all_histoire_debut_embedding), np.array(all_histoire_noise_debut),
                    np.array(all_histoire_fin_embedding), np.array(all_histoire_noise_fin)], [zeros, zeros, zeros,
                                                                                              zeros,
                                                                                              zeros, ones, zeros, ones]
        return [np.array(all_histoire_debut_embedding), np.array(all_histoire_noise_debut),
                np.array(all_histoire_fin_embedding), np.array(all_histoire_noise_fin)], [zeros, zeros, zeros, ones]

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
        label1 = []
        label2 = []
        for b in batch:
            histoire_debut = " ".join(b[3]) + " " + " ".join(b[4])
            histoire_noise_debut = self.add_noise(histoire_debut)
            histoire_fin = " ".join(b[3]) + " " + " ".join(b[5])
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
        zeros = np.zeros(len(batch))
        label1 = np.array(label1)
        label2 = np.array(label2)
        if self.use_pretrained_models:
            return [np.array(all_histoire_debut_embedding),
                    np.array(all_histoire_noise_debut),
                    np.array(all_histoire_fin_embedding),
                    np.array(all_histoire_noise_fin)], [zeros, zeros, zeros, zeros, label1, label2, label1, label2]
        return [np.array(all_histoire_debut_embedding),
                np.array(all_histoire_noise_debut),
                np.array(all_histoire_fin_embedding),
                np.array(all_histoire_noise_fin)], [zeros, zeros, label1, label2]


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
    def __init__(self, layer1, layer2, layer3, name=None):
        self.layer2 = layer2
        self.layer1 = layer1
        self.layer3 = layer3
        if name is not None:
            self.layer3.name = name

    def __call__(self, x):
        l1 = BatchNormalization()(Dropout(0.3)(LeakyReLU()(self.layer1(x))))
        l2 = BatchNormalization()(Dropout(0.3)(LeakyReLU()(self.layer2(l1))))
        return BatchNormalization()(self.layer3(l2))
