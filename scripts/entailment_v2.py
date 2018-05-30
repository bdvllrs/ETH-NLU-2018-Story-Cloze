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
from utils import SNLIDataloader
from nltk import word_tokenize
from utils import Dataloader
from scripts import DefaultScript


class Script(DefaultScript):

    slug = 'entailment'

    def train(self):
        main(self.config)

    def test(self):
        testing_set = Dataloader(self.config, testing_data=True)
        testing_set.load_dataset('data/test.bin')

        testing_set.load_vocab('data/default.voc', self.config.vocab_size)
        test(self.config, testing_set)


class Preprocess:
    def __init__(self, sent2vec):
        self.sent2vec = sent2vec

    def __call__(self, line):
        # label = [entailment, neutral, contradiction]
        label = [1]
        if line['gold_label'] == 'contradiction':
            label = [0]
        elif line['gold_label'] == 'neutral':
            label = [1]
        sentence1 = list(self.sent2vec.embed_sentence(' '.join(word_tokenize(line['sentence1']))))
        sentence2 = list(self.sent2vec.embed_sentence(' '.join(word_tokenize(line['sentence2']))))
        output = [label, sentence1, sentence2]
        return output


class PreprocessTest:
    """
    Preprocess to apply to the dataset
    """

    def __init__(self, sent2vec_model):
        self.sent2vec_model = sent2vec_model

    def __call__(self, word_to_index, sentence):
        sentence = ' '.join(sentence)
        return sentence


def output_fn(_, batch):
    batch = np.array(batch)
    return [np.array(list(batch[:, 1])), np.array(list(batch[:, 2]))], np.array(list(batch[:, 0]))[:, 0]


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
            sentence = batch[b][3]
            sentence_batch.append(self.sent2vec.embed_sentence(sentence))
            ending_1.append(self.sent2vec.embed_sentence(batch[b][4]))
            ending_2.append(self.sent2vec.embed_sentence(batch[b][5]))
        endings = ending_2[:]
        correct_ending = data.label
        label = np.array(correct_ending) - 1
        final_label = []
        # correct ending if 1 --> if 2 true get 2 - 1 = 1, if 1 true get 1 - 1 = 0
        if random.random() > 0.5:
            endings = ending_1[:]
            label = 1 - label
        for b in range(len(label)):
            final_label.append([1] if label[b] == 1 else [0])
        # Return what's needed for keras
        return [np.array(sentence_batch), np.array(endings)], np.array(final_label)


def model(config):
    dense_layer_1 = keras.layers.Dense(500, activation='relu')
    dense_layer_2 = keras.layers.Dense(100, activation='relu')
    dense_layer_3 = keras.layers.Dense(1, activation='sigmoid')

    sentence_1 = keras.layers.Input(shape=(config.sent2vec.embedding_size,))
    sentence_2 = keras.layers.Input(shape=(config.sent2vec.embedding_size,))
    # Graph
    inputs = keras.layers.Concatenate()([sentence_1, sentence_2])
    # inputs = sentiments
    output = keras.layers.Dropout(0.3)(dense_layer_1(inputs))
    output = keras.layers.Dropout(0.3)(dense_layer_2(output))
    output = dense_layer_3(output)

    # Model
    model = keras.models.Model(inputs=[sentence_1, sentence_2], outputs=[output])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    return model


def main(config):
    import sent2vec
    assert config.sent2vec.model is not None, "Please add sent2vec_model config value."
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(config.sent2vec.model)

    preprocess_fn = Preprocess(sent2vec_model)

    train_set = SNLIDataloader('data/snli_1.0/snli_1.0_train.jsonl')
    train_set.set_preprocess_fn(preprocess_fn)
    train_set.set_output_fn(output_fn)
    dev_set = SNLIDataloader('data/snli_1.0/snli_1.0_dev.jsonl')
    dev_set.set_preprocess_fn(preprocess_fn)
    dev_set.set_output_fn(output_fn)
    test_set = SNLIDataloader('data/snli_1.0/snli_1.0_test.jsonl')

    generator_training = train_set.get_batch(config.batch_size, config.n_epochs)
    generator_dev = dev_set.get_batch(config.batch_size, config.n_epochs)

    keras_model = model(config)

    verbose = 0 if not config.debug else 1
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Callbacks
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/' + timestamp + '-entailmentv2/', histogram_freq=0,
                                              batch_size=config.batch_size,
                                              write_graph=False,
                                              write_grads=True)

    model_path = os.path.abspath(
        os.path.join(os.curdir, './builds/' + timestamp))
    model_path += '-entailmentv2_checkpoint_epoch-{epoch:02d}.hdf5'

    saver = keras.callbacks.ModelCheckpoint(model_path,
                                            monitor='val_loss', verbose=verbose, save_best_only=True)

    keras_model.fit_generator(generator_training, steps_per_epoch=100,
                              epochs=config.n_epochs,
                              verbose=verbose,
                              validation_data=generator_dev,
                              validation_steps=len(test_set) / config.batch_size,
                              callbacks=[tensorboard, saver])


def test(config, testing_set):
    import sent2vec
    assert config.sent2vec.model is not None, "Please add sent2vec_model config value."
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(config.sent2vec.model)

    preprocess_fn = PreprocessTest(sent2vec_model)
    testing_set.set_preprocess_fn(preprocess_fn)

    output_fn_test = OutputFnTest(sent2vec_model, config)

    testing_set.set_output_fn(output_fn_test)

    generator_testing = testing_set.get_batch(config.batch_size, config.n_epochs, random=True)

    keras_model = keras.models.load_model(
        './builds/leonhard/2018-05-19 22:33:08-entailmentv2_checkpoint_epoch-1810.hdf5')

    verbose = 0 if not config.debug else 1

    # test_batch = next(generator_testing)
    loss = keras_model.evaluate_generator(generator_testing, steps=len(testing_set) / config.batch_size,
                                          verbose=verbose)
    print(loss)
