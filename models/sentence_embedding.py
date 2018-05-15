import tensorflow as tf


class SentenceEmbedding:
    def __init__(self, config):
        self.embedding_size = config.sent2vec.embedding_size
        self.batch_size = config.batch_size
        self.num_sentences = 4
        self.learning_rate = config.learning_rate
        self.optimizer = None
        self.label = None
        self.cross_entropy = None

    def __call__(self):
        with tf.variable_scope("sentence_embedding"):
            # input the sentence embedding
            # batch size x embedding_size
            # Only keep last sentence
            self.last_sentence = tf.placeholder(tf.float64, (self.batch_size, self.embedding_size),
                                                name="last-sentence")
            self.sentiments = tf.placeholder(tf.float64,
                                             (self.batch_size, 2 * 5),
                                             name="sentiment")  # pos and neg for last sentence and ending
            self.ending = tf.placeholder(tf.float64, (self.batch_size, self.embedding_size), "ending")
            sem_embeddings = tf.concat((self.last_sentence + self.ending, self.sentiments), axis=1)
            hidden1 = tf.layers.dense(sem_embeddings, 300, activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      bias_initializer=tf.contrib.layers.xavier_initializer())
            hidden2 = tf.layers.dense(hidden1, 100, activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      bias_initializer=tf.contrib.layers.xavier_initializer())
            self.output = tf.squeeze(tf.layers.dense(hidden2, 1, activation=tf.nn.relu,
                                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                     bias_initializer=tf.contrib.layers.xavier_initializer()),
                                     name="output")
            return self.output

    def optimize(self):
        with tf.variable_scope("sentence_embedding/optimize"):
            self.label = tf.placeholder(tf.float64, self.batch_size,
                                        name="label")
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.output)
            self.cross_entropy = tf.reduce_mean(cross_entropy, name="cross-entropy")
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer = optimizer.minimize(self.cross_entropy, name="optimizer")
