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
from keras.utils.np_utils import to_categorical
import sent2vec
import numpy as np
from utils import SNLIDataloader
from nltk import word_tokenize
from scripts import DefaultScript


class Script(DefaultScript):

    slug = 'type_translation'

    def train(self):
        main(self.config)


class Preprocess:
    def __init__(self, sent2vec):
        self.sent2vec = sent2vec

    def __call__(self, line):
        sentence1 = list(self.sent2vec.embed_sentence(' '.join(word_tokenize(line['sentence1']))))
        sentence2 = ' '.join(word_tokenize(line['sentence2']))
        output = [sentence1, sentence2]
        return output


def output_fn(word_to_index, batch):
    batch = np.array(batch)
    num_classes = len(word_to_index.keys())
    input_sentences = np.expand_dims(np.array(list(batch[:, 0])), 1)
    output_sentences = batch[:, 1]
    output = []
    max_size = 0
    for b in range(len(output_sentences)):
        words = output_sentences[b].split(' ')
        sentence = []
        for word in words:
            word = word.lower()
            label = word_to_index[word] if word in word_to_index.keys() else word_to_index['<unk>']
            sentence.append(label)
        if len(sentence) > max_size:
            max_size = len(sentence)
        output.append(sentence)
    for b in range(len(output_sentences)):
        output[b] += [word_to_index['<pad>']] * (max_size - len(output[b]))
        for k, word in enumerate(output[b]):
            output[b][k] = to_categorical(word, num_classes=num_classes)
    return input_sentences, np.array(output)


def model(config):
    dense_layer_1 = keras.layers.Dense(500, activation='relu')
    dense_layer_2 = keras.layers.Dense(500, activation='relu')
    dense_layer_3 = keras.layers.Dense(config.vocab_size, activation='softmax')
    gru_layer = keras.layers.GRU(500, return_sequences=True)

    sentence = keras.layers.Input(shape=(None, config.sent2vec.embedding_size,))
    # inputs = sentiments
    sequences = gru_layer(sentence)
    output = keras.layers.Dropout(0.3)(dense_layer_1(sequences))
    output = keras.layers.Dropout(0.3)(dense_layer_2(output))
    output = dense_layer_3(output)

    # Model
    model = keras.models.Model(inputs=sentence, outputs=output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    return model


def main(config):
    assert config.sent2vec.model is not None, "Please add sent2vec_model config value."
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(config.sent2vec.model)

    preprocess_fn = Preprocess(sent2vec_model)

    train_set = SNLIDataloader('data/snli_1.0/snli_1.0_train.jsonl')
    train_set.load_vocab('./data/snli_vocab.dat', config.vocab_size)
    train_set.set_preprocess_fn(preprocess_fn)
    train_set.set_output_fn(output_fn)
    dev_set = SNLIDataloader('data/snli_1.0/snli_1.0_dev.jsonl')
    dev_set.load_vocab('./data/snli_vocab.dat', config.vocab_size)
    dev_set.set_preprocess_fn(preprocess_fn)
    dev_set.set_output_fn(output_fn)
    test_set = SNLIDataloader('data/snli_1.0/snli_1.0_test.jsonl')

    generator_training = train_set.get_batch(config.batch_size, config.n_epochs, only_contradiction=True)
    generator_dev = dev_set.get_batch(config.batch_size, config.n_epochs, only_contradiction=True)

    keras_model = model(config)

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
                                            monitor='val_acc', verbose=verbose, save_best_only=True)

    keras_model.fit_generator(generator_training, steps_per_epoch=5,
                              epochs=config.n_epochs,
                              verbose=verbose,
                              validation_data=generator_dev,
                              validation_steps=len(test_set) / config.batch_size,
                              callbacks=[tensorboard, saver])
