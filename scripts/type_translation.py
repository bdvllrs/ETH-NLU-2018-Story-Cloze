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
from keras.utils import to_categorical
import numpy as np
from utils import SNLIDataloader
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
            "elmo"]


def preprocess_fn(line):
    output = [line['sentence1'], line['sentence2']]
    return output


def output_fn(word_to_index, batch):
    num_classes = len(word_to_index.keys())
    input_sentences = []
    output_sentences = []
    for b in batch:
        input_sentences.append(b[0])
        output_sentences.append(b[1])
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


class ModelLoss:
    def __init__(self, elmo_embedding):
        self.elmo_embedding = elmo_embedding
        self.__name__ = 'model_loss'

    def __call__(self, y_true, y_pred):
        emb_true = self.elmo_embedding(y_true)
        emb_pred = self.elmo_embedding(y_pred)
        y_true = K.l2_normalize(emb_true, axis=-1)
        y_pred = K.l2_normalize(emb_pred, axis=-1)
        return -K.sum(y_true * y_pred, axis=-1)


def model(sess, config):
    if config.debug:
        print('Importing Elmo module...')
    if config.hub.is_set("cache_dir"):
        os.environ['TFHUB_CACHE_DIR'] = config.hub.cache_dir

    elmo_model = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
    if config.debug:
        print('Imported.')

    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    elmo_emb_fn = ElmoEmbedding(elmo_model)

    elmo_embeddings = keras.layers.Lambda(elmo_emb_fn, output_shape=(None, 1024))

    gru_layer = keras.layers.GRU(700, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)
    dense_layer_1 = keras.layers.Dense(500, activation='relu')
    dense_layer_2 = keras.layers.Dense(500, activation='relu')
    dense_layer_3 = keras.layers.Dense(config.vocab_size, activation='softmax')

    sentence = keras.layers.Input(shape=(1,), dtype="string")
    sentence_emb = elmo_embeddings(sentence)

    # inputs = sentiments
    sequences = gru_layer(sentence_emb)
    output = keras.layers.Dropout(0.3)(dense_layer_1(sequences))
    output = keras.layers.Dropout(0.3)(dense_layer_2(output))
    output = dense_layer_3(output)

    # Model
    loss = ModelLoss(elmo_emb_fn)
    model = keras.models.Model(inputs=sentence, outputs=output)
    model.compile(optimizer="adam", loss=loss, metrics=['accuracy'])
    return model


def main(config):
    train_set = SNLIDataloader('data/snli_1.0/snli_1.0_train.jsonl')
    train_set.load_vocab('./data/snli_vocab.dat', config.vocab_size)
    train_set.set_preprocess_fn(preprocess_fn)
    train_set.set_output_fn(output_fn)
    dev_set = SNLIDataloader('data/snli_1.0/snli_1.0_dev.jsonl')
    dev_set.load_vocab('./data/snli_vocab.dat', config.vocab_size)
    dev_set.set_preprocess_fn(preprocess_fn)
    dev_set.set_output_fn(output_fn)
    # test_set = SNLIDataloader('data/snli_1.0/snli_1.0_test.jsonl')

    generator_training = train_set.get_batch(config.batch_size, config.n_epochs, only_contradiction=True)
    generator_dev = dev_set.get_batch(config.batch_size, config.n_epochs, only_contradiction=True)

    # Initialize tensorflow session
    sess = tf.Session()
    K.set_session(sess)  # Set to keras backend

    keras_model = model(sess, config)

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
                              validation_steps=len(dev_set) / config.batch_size,
                              callbacks=[tensorboard, saver])
