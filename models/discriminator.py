import tensorflow as tf


def discriminator(batch_size, vocab_size, embedding_size, hidden_size, num_rnns=2):
    """
    :param batch_size:
    :param vocab_size:
    :param embedding_size:
    :param hidden_size:
    :param num_rnns:
    """
    with tf.variable_scope("discriminator"):
        x = tf.placeholder(tf.int32, (batch_size, None), name="x")

        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            word_embeddings = tf.get_variable("word_embeddings",
                                              [vocab_size, embedding_size], dtype=tf.float32)

        # Shape: batch size x sequence length x embedding size
        inputs = tf.nn.embedding_lookup(word_embeddings, x, name="input")
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=[rnn_cell(hidden_size) for _ in range(num_rnns)], cells_bw=[rnn_cell(hidden_size) for _ in range(num_rnns)],
            inputs=inputs, dtype=tf.float32)
        summed_output = tf.reduce_sum(outputs, 1)
        # Gives an output between 0 and 1
        # representing the probability of the last sentence being correct
        output = tf.contrib.layers.fully_connected(summed_output, 1, activation_fn=tf.sigmoid)
        return output
