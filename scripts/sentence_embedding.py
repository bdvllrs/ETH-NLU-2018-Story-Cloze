import os
import random
import datetime
import keras
import numpy as np
import sent2vec
from utils import Dataloader
from scripts import DefaultScript


class Script(DefaultScript):

    slug = 'sentence_embedding'

    def train(self):
        training_set = Dataloader(self.config)
        training_set.load_dataset('./data/train.bin')
        training_set.load_vocab('./default.voc', self.config.vocab_size)

        testing_set = Dataloader(self.config, testing_data=True)
        testing_set.load_dataset('data/test.bin')
        testing_set.load_vocab('./default.voc', self.config.vocab_size)

        main(self.config, training_set, testing_set)


class Preprocess:
    """
    Preprocess to apply to the dataset
    """

    def __init__(self, sent2vec_model):
        self.sent2vec_model = sent2vec_model

    def __call__(self, word_to_index, sentence):
        sentence = self.sent2vec_model.embed_sentence(' '.join(sentence))
        return sentence


def output_fn_train(data):
    batch = np.array(data.batch)
    last_sentences = batch[:, 3, :]
    endings = batch[:, 4, :]
    sentiments = np.array(data.sentiments)
    label = []
    # Randomly choose a bad ending
    for i in range(len(batch)):
        # Only 50% of the time
        if random.random() > 0.5:
            k = random.randint(0, len(data.dataloader) - 1)
            new_data = data.dataloader.get(k, raw=True)
            new_batch = np.array(new_data.batch)
            endings[i] = new_batch[0, 4, :]
            label.append(0)
            sentiments[i] = np.array(new_data.sentiments)[0, :]
        else:
            label.append(1)
    # Return what's needed for keras
    return [last_sentences, endings, sentiments], np.array(label)


def output_fn_test(data):
    batch = np.array(data.batch)
    last_sentences = batch[:, 3, :]
    ending_1 = batch[:, 4, :]
    ending_2 = batch[:, 5, :]
    sentiments = np.array(data.sentiments)[:, :6]
    correct_ending = data.label
    endings = ending_2[:]
    sentiment_1 = sentiments[:, :5]
    sentiment_2 = np.concatenate((sentiments[:, :4], sentiments[:, 5:6]), axis=1)
    sentiments = sentiment_2
    # correct ending if 1 --> if 2 true get 2 - 1 = 1, if 1 true get 1 - 1 = 0
    label = np.array(correct_ending) - 1
    if random.random() > 0.5:
        endings = ending_1[:]
        label = 1 - label
        sentiments = sentiment_1
    # Return what's needed for keras
    return [last_sentences, endings, sentiments], label


def keras_model(config):
    # Layers
    dense_layer_1 = keras.layers.Dense(500, activation='relu')
    dense_layer_2 = keras.layers.Dense(100, activation='relu')
    dense_layer_3 = keras.layers.Dense(1, activation='sigmoid')

    # Inputs
    last_sentence = keras.layers.Input(shape=(config.sent2vec.embedding_size,))
    ending = keras.layers.Input(shape=(config.sent2vec.embedding_size,))
    sentiments = keras.layers.Input(shape=(5,))

    # Graph
    inputs = keras.layers.Concatenate()([last_sentence, ending, sentiments])
    # inputs = sentiments
    output = keras.layers.Dropout(0.3)(dense_layer_1(inputs))
    output = keras.layers.Dropout(0.3)(dense_layer_2(output))
    output = dense_layer_3(output)

    # Model
    model = keras.models.Model(inputs=[last_sentence, ending, sentiments], outputs=[output])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    return model


def main(config, training_set, testing_set):
    assert config.sent2vec.model is not None, "Please add sent2vec_model config value."
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(config.sent2vec.model)

    preprocess_fn = Preprocess(sent2vec_model)

    training_set.set_preprocess_fn(preprocess_fn)
    testing_set.set_preprocess_fn(preprocess_fn)

    training_set.set_output_fn(output_fn_train)
    testing_set.set_output_fn(output_fn_test)

    generator_training = training_set.get_batch(config.batch_size, config.n_epochs, random=True)
    generator_testing = testing_set.get_batch(config.batch_size, config.n_epochs, random=True)

    cloze_model = keras_model(config)

    verbose = 0 if not config.debug else 1
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Callbacks
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/sentence-embedding-' + timestamp + '/', histogram_freq=0,
                                              batch_size=config.batch_size,
                                              write_graph=False,
                                              write_grads=True)

    model_path = os.path.abspath(
        os.path.join(os.curdir, './builds/' + timestamp))
    model_path += '-sentence-embedding_checkpoint_epoch-{epoch:02d}.hdf5'

    saver = keras.callbacks.ModelCheckpoint(model_path,
                                            monitor='val_loss', verbose=verbose, save_best_only=True)

    cloze_model.fit_generator(generator_training, steps_per_epoch=len(training_set) / config.batch_size,
                              epochs=config.n_epochs,
                              verbose=verbose,
                              validation_data=generator_testing,
                              validation_steps=len(testing_set) / config.batch_size,
                              callbacks=[tensorboard, saver])
