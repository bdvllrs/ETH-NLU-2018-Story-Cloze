import numpy as np
import tensorflow as tf
from nltk import word_tokenize


class Scheduler:
    def __init__(self, batch_size, vocab_size, embedding_size, hidden_size):
        """
        :param batch_size:
        :param vocab_size:
        :param embedding_size:
        :param hidden_size:
        """
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_sentences = 5
        self.labels = None
        self.optimizer = None
        self.mse = None
        self.probabilities = None

    def __call__(self):
        """
        Tries to order the sequences in the right order
        """
        with tf.variable_scope("scheduler"):
            # batch size x number of sentences x sequence length
            self.x = tf.placeholder(tf.int32, (self.batch_size, self.num_sentences, None), name="x")

            with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
                self.word_embeddings = tf.get_variable("word_embeddings",
                                                           [self.vocab_size, self.embedding_size], dtype=tf.float32)
            # Shape: batch size x sequence length x embedding size
            inputs = tf.nn.embedding_lookup(self.word_embeddings, self.x, name="input")
            inputs_flattened = tf.reshape(inputs, (self.batch_size * self.num_sentences, -1, self.embedding_size))
            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell
            outputs_flattened, _ = tf.nn.dynamic_rnn(rnn_cell(self.hidden_size), inputs_flattened, dtype=tf.float32)
            outputs = tf.reshape(outputs_flattened,
                                 (self.batch_size, self.num_sentences, -1, self.hidden_size))
            summed_output = tf.expand_dims(tf.reduce_sum(outputs, 2), 3)
            conv1 = tf.layers.conv2d(
                inputs=summed_output,
                filters=5,
                kernel_size=[2, 5],
                padding='same',
                activation=tf.nn.relu
            )
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 5], strides=[1, 2])
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=1,
                kernel_size=[2, 5],
                padding="same",
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 5], strides=2)
            flattened_sentences = tf.reshape(pool2, (self.batch_size, -1))
            # Gives an output between 0 and 1
            # representing the probability of the last sentence being correct
            output = tf.contrib.layers.fully_connected(flattened_sentences, self.num_sentences-3)
            # output = tf.reshape(output, (self.batch_size, self.num_sentences-3))
            order_probability = tf.nn.softmax(output, name='order_probability')
            self.probabilities = order_probability
            return order_probability

    def optimize(self, learning_rate):
        with tf.variable_scope("scheduler/optimize"):
            self.labels = tf.placeholder(tf.int32, (self.batch_size, self.num_sentences-3), name="label")
            training_vars = tf.trainable_variables()
            mse_total = tf.losses.mean_squared_error(self.labels, self.probabilities)
            self.mse = tf.identity(mse_total, name="mse")
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.mse, training_vars), 5)  # Max gradient of 5
            optimizer = tf.train.AdamOptimizer(learning_rate)
            optimizer.apply_gradients(zip(grads, training_vars))
            self.optimizer = optimizer.minimize(self.mse, name="optimizer")


def scheduler(batch_size, vocab_size, embedding_size, hidden_size):
    """
    Tries to order the sequences in the right order
    :param batch_size:
    :param vocab_size:
    :param embedding_size:
    :param hidden_size:
    """
    num_sentences = 5
    with tf.variable_scope("scheduler"):
        # batch size x number of sentences x sequence length
        x = tf.placeholder(tf.int32, (batch_size, num_sentences, None), name="x")

        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            word_embeddings = tf.get_variable("word_embeddings",
                                              [vocab_size, embedding_size], dtype=tf.float32)

        # Shape: batch size x sequence length x embedding size
        inputs = tf.nn.embedding_lookup(word_embeddings, x, name="input")
        inputs_flattened = tf.reshape(inputs, (batch_size * num_sentences, -1, embedding_size))

        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell
        outputs_flattened, _ = tf.nn.dynamic_rnn(rnn_cell(hidden_size), inputs_flattened, dtype=tf.float32)
        outputs = tf.reshape(outputs_flattened,
                             (batch_size, num_sentences, -1, hidden_size))
        summed_output = tf.expand_dims(tf.reduce_sum(outputs, 2), 3)
        conv1 = tf.layers.conv2d(
            inputs=summed_output,
            filters=5,
            kernel_size=[2, 5],
            padding='same',
            activation=tf.nn.relu
        )
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 5], strides=[1, 2])
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=1,
            kernel_size=[2, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 5], strides=2)
        flattened_sentences = tf.reshape(pool2, (batch_size, -1))
        # Gives an output between 0 and 1
        # representing the probability of the last sentence being correct
        output = tf.contrib.layers.fully_connected(flattened_sentences, num_sentences * num_sentences)
        output = tf.reshape(output, (batch_size, num_sentences, num_sentences))
        order_probability = tf.nn.softmax(output, name='order_probability')
        return word_embeddings, order_probability


def scheduler_optimize(probabilities, learning_rate, batch_size):
    """
    Optimization function for the scheduler
    :return:
    """
    num_sentences = 5
    with tf.variable_scope("scheduler/optimize"):
        labels = tf.placeholder(tf.int32, (batch_size, num_sentences, num_sentences), name="label")
        training_vars = tf.trainable_variables()
        mse_total = tf.losses.mean_squared_error(labels, probabilities)
        mse = tf.identity(mse_total, name="mse")
        grads, _ = tf.clip_by_global_norm(tf.gradients(mse, training_vars), 5)  # Max gradient of 5
        optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer.apply_gradients(zip(grads, training_vars))
        opt = optimizer.minimize(mse, name="optimizer")
        return opt, mse


def scheduler_preprocess(word_to_index, line):
    sentences = line[2:]  # remove the 2 first cols id and title
    tokenized_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = word_tokenize(sentence)
        # replace unknown by <unk>
        sentence = list(map(lambda word: word if word in word_to_index.keys() else '<unk>', sentence))
        # replace words by tokens
        sentence = list(map(lambda word: word_to_index[word], sentence))
        tokenized_sentences.append(sentence)
    return tokenized_sentences


def scheduler_get_labels(batch):
    """
    Shuffles the sentences in the story.
    :param batch:
    :return: (batch, labels) labels of size (batch size x # sentences)
    """
    new_batch = np.copy(batch)
    labels = np.zeros((len(batch), len(batch[0])-3))
    for k in range(len(batch)):
        sentences = list(range(3, len(batch[k])))
        np.random.shuffle(sentences)
        for i in range(3):
            new_batch[k][i] = batch[k][i]
        for i in range(3, len(batch[k])):
            new_batch[k][i] = batch[k][sentences[i-3]]
            labels[k][i-3] = 1 if sentences[i-3] == len(batch[k])-1 else 0
    return new_batch, labels


if __name__ == '__main__':
    scheduler(10, 20000, 100, 100)
