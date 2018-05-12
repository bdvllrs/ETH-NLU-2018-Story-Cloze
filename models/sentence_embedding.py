import tensorflow as tf


class SentenceEmbedding:
    def __init__(self, config):
        self.embedding_size = config.sent2vec.embedding_size
        self.batch_size = config.batch_size
        self.num_sentences = 4
        self.learning_rate = config.learning_rate
        self.optimizer = None
        self.label = None
        self.distance_sum = None

    def __call__(self):
        with tf.variable_scope("sentence_embedding"):
            # input the sentence embedding
            # batch size x number of sentences x embedding_size
            self.x = tf.placeholder(tf.float64, (self.batch_size, self.num_sentences, self.embedding_size), name="x")
            self.label = tf.placeholder(tf.float64, (self.batch_size, self.embedding_size), "label")
            hidden1 = tf.layers.dense(self.x, 100, activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      bias_initializer=tf.contrib.layers.xavier_initializer())
            hidden2 = tf.layers.dense(hidden1, 100, activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      bias_initializer=tf.contrib.layers.xavier_initializer())
            self.output = tf.layers.dense(hidden2, 50, activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      bias_initializer=tf.contrib.layers.xavier_initializer(), name="output")
            input_mean = tf.reduce_mean(self.output, axis=1)
            self.distance = tf.identity(tf.norm(self.label - input_mean, axis=1), name="distance")
            self.distance_sum = tf.reduce_sum(self.distance, name="distance_sum")
            return self.output

    def optimize(self):
        with tf.variable_scope("sentence_embedding/optimize"):
            training_vars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.distance_sum, training_vars), 5)  # Max gradient of 5
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            optimizer.apply_gradients(zip(grads, training_vars))
            self.optimizer = optimizer.minimize(self.distance_sum, name="optimizer")
