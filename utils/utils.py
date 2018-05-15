import datetime
from gensim import models
import tensorflow as tf
import numpy as np
from tqdm import tqdm


def load_embedding(session, vocab, emb, path, dim_embedding, vocab_size):
    """
      session        Tensorflow session object
      vocab          A dictionary mapping token strings to vocabulary IDs
      emb            Embedding tensor of shape vocabulary_size x dim_embedding
      path           Path to embedding file
      dim_embedding  Dimensionality of the external embedding.
    """

    print("Loading external embeddings from %s" % path)

    model = models.KeyedVectors.load_word2vec_format(path, binary=False)
    external_embedding = np.zeros(shape=(vocab_size, dim_embedding))
    matches = 0

    for tok, idx in vocab.items():
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)

    print("%d words out of %d could be loaded" % (matches, vocab_size))

    pretrained_embeddings = tf.placeholder(tf.float32, [None, None])
    assign_op = emb.assign(pretrained_embeddings)
    session.run(assign_op, {pretrained_embeddings: external_embedding})  # here, embeddings are actually set


def train_test(config, training_set, testing_set, test_fn, train_fn):
    nthreads_intra = config.nthreads // 2
    nthreads_inter = config.nthreads - config.nthreads // 2

    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=nthreads_inter,
                                          intra_op_parallelism_threads=nthreads_intra)) as sess:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        train_writer = tf.summary.FileWriter('./logs/' + timestamp + '/train/', sess.graph)
        test_writer = tf.summary.FileWriter('./logs/' + timestamp + '/test/', sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for epoch in range(config.n_epochs):
            if config.debug:
                print("Epoch", epoch)
            if not epoch % config.test_every:
                if config.debug:
                    print('Testing...')
                    progress_bar = tqdm(total=len(testing_set))
                # Testing phase
                success = 0
                total = 0
                for k in range(0, len(testing_set), config.batch_size):
                    if k + config.batch_size < len(testing_set):
                        new_total, new_success = test_fn(config, testing_set, sess, epoch, k)
                        success += new_success
                        total += new_total
                        if config.debug:
                            progress_bar.update(config.batch_size)
                accuracy = float(success) / float(total)
                accuracy_summary = tf.Summary()
                accuracy_summary.value.add(tag='accuracy', simple_value=accuracy)
                test_writer.add_summary(accuracy_summary, epoch)
                print("Testing:", accuracy)
                if config.debug:
                    progress_bar.close()
            if config.debug:
                progress_bar = tqdm(total=len(training_set))
            for k in range(0, len(training_set), config.batch_size):
                if k + config.batch_size < len(training_set):
                    summary_op = tf.summary.merge_all()
                    train_fn(config, training_set, sess, epoch, k, summary_op, train_writer)
                    if config.debug:
                        progress_bar.update(config.batch_size)
            if config.debug:
                progress_bar.close()
            training_set.shuffle_lines()
            if not epoch % config.save_model_every:
                model_path = './builds/' + timestamp + '/model'
                saver.save(sess, model_path, global_step=epoch)
