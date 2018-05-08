import numpy as np
import tensorflow as tf
from nltk import word_tokenize


def scheduler(batch_size, vocab_size, embedding_size, hidden_size, num_rnns=2):
    """
    Tries to order the sequences in the right order
    :param batch_size:
    :param vocab_size:
    :param embedding_size:
    :param hidden_size:
    :param num_rnns:
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
        outputs_flattened, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=[rnn_cell(hidden_size) for _ in range(num_rnns)], cells_bw=[rnn_cell(hidden_size) for _ in range(num_rnns)],
            inputs=inputs_flattened, dtype=tf.float32)
        outputs = tf.reshape(outputs_flattened, (batch_size, num_sentences, -1, 2 * hidden_size))  # *2 because bi directional
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
        return order_probability


def scheduler_optimize(probabilities, learning_rate, batch_size):
    """
    Optimization function for the scheduler
    :return:
    """
    num_sentences = 5
    with tf.variable_scope("scheduler/optimize"):
        labels = tf.placeholder(tf.int32, (batch_size, num_sentences), name="label")
        training_vars = tf.trainable_variables()
        cross_entropy_total = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=probabilities)
        cross_entropy = tf.identity(cross_entropy_total, name="cross_entropy")
        grads, _ = tf.clip_by_global_norm(tf.gradients(cross_entropy, training_vars), 5)  # Max gradient of 5
        optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer.apply_gradients(zip(grads, training_vars))
        opt = optimizer.minimize(cross_entropy, name="optimizer")
        return opt, cross_entropy


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
    labels = np.zeros((len(batch), len(batch[0])))
    for k in range(len(batch)):
        sentences = list(range(len(batch[k])))
        np.random.shuffle(sentences)
        for i in range(len(batch[k])):
            new_batch[k][i] = batch[k][sentences[i]]
            labels[k][i] = sentences[i]
    return new_batch, labels


if __name__ == '__main__':
    scheduler(10, 20000, 100, 100)
