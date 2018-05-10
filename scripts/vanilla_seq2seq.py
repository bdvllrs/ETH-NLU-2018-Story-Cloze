import datetime
from tqdm import tqdm
import tensorflow as tf
from models import VanillaSeq2SeqEncoder, scheduler_preprocess, scheduler_get_labels
from utils import load_embedding


def main(config, training_set, testing_set):
    training_set.set_preprocess_fn(scheduler_preprocess)
    training_set.set_special_tokens(['<pad>', '<unk>'])

    scheduler_model = VanillaSeq2SeqEncoder(config.batch_size, config.vocab_size, config.embedding_size, config.hidden_size)
    _ = scheduler_model()
    scheduler_model.optimize(config.learning_rate)

    tf.summary.scalar("cost", scheduler_model.mse)

    nthreads_intra = config.nthreads // 2
    nthreads_inter = config.nthreads - config.nthreads // 2

    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=nthreads_inter,
                                          intra_op_parallelism_threads=nthreads_intra)) as sess:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer = tf.summary.FileWriter('./logs/' + timestamp, sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # Load word2vec pretrained embeddings
        load_embedding(sess, training_set.word_to_index, scheduler_model.word_embeddings, config.embedding_path,
                       config.embedding_size, config.vocab_size)

        for epoch in range(config.n_epochs):
            for k in tqdm(range(0, len(training_set), config.batch_size)):
                if k + config.batch_size < len(training_set):
                    summary_op = tf.summary.merge_all()

                    batch = training_set.get(k, config.batch_size, random=True)
                    shuffled_batch, labels = scheduler_get_labels(batch)
                    probabilities, _, computed_mse, summary = sess.run(
                        ['scheduler/order_probability:0', 'scheduler/optimize/optimizer',
                         'scheduler/optimize/mse:0', summary_op],
                        {'scheduler/x:0': shuffled_batch,
                         'scheduler/optimize/label:0': labels})
                    writer.add_summary(summary, epoch * len(training_set) + k)
                    if not epoch % config.save_model_every:
                        model_path = './builds/' + timestamp
                        saver.save(sess, model_path, global_step=epoch)
            training_set.shuffle_lines()
            if not epoch % config.save_model_every:
                model_path = './builds/' + timestamp + '/model'
                saver.save(sess, model_path, global_step=epoch)
