import datetime
import tensorflow as tf
from tqdm import tqdm
from models import SentenceEmbedding
import sent2vec


class Preprocess:
    def __init__(self, sent2vec_model):
        self.sent2vec_model = sent2vec_model

    def __call__(self, word_to_index, sentence):
        # Get sentence level embedding with sent2vec
        sentence = self.sent2vec_model.embed_sentence(' '.join(sentence))
        return sentence


def get_labels(batch):
    return batch[:, :4, :], batch[:, 4, :]


def main(config, training_set, testing_set):
    assert config.sent2vec.model is not None, "Please add sent2vec_model config value."
    model = sent2vec.Sent2vecModel()
    model.load_model(config.sent2vec.model)
    preprocess = Preprocess(model)
    training_set.set_preprocess_fn(preprocess)
    testing_set.set_preprocess_fn(preprocess)

    sentence_embedding = SentenceEmbedding(config)
    sentence_embedding()
    sentence_embedding.optimize()

    tf.summary.scalar("distance", sentence_embedding.distance_sum)

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
                        batch_endings1, batch_endings2, correct_ending = testing_set.get(k, config.batch_size,
                                                                                         random=True)
                        total += config.batch_size
                        shuffled_batch1, labels1 = get_labels(batch_endings1)
                        shuffled_batch2, labels2 = get_labels(batch_endings2)
                        distances1 = sess.run(
                            'sentence_embedding/distance:0',
                            {'sentence_embedding/x:0': shuffled_batch1,
                             'sentence_embedding/label:0': labels1})
                        distances2 = sess.run(
                            'sentence_embedding/distance:0',
                            {'sentence_embedding/x:0': shuffled_batch2,
                             'sentence_embedding/label:0': labels2})
                        for b in range(config.batch_size):
                            if distances1[b] < distances2[b]:
                                if correct_ending[b] == 0:
                                    success += 1
                            else:
                                if correct_ending[b] == 1:
                                    success += 1
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

                    batch = training_set.get(k, config.batch_size, random=True)
                    shuffled_batch, labels = get_labels(batch)
                    distances, _, summary = sess.run(
                        ['sentence_embedding/distance_sum:0', 'sentence_embedding/optimize/optimizer', summary_op],
                        {'sentence_embedding/x:0': shuffled_batch,
                         'sentence_embedding/label:0': labels})
                    print(distances)
                    if config.debug:
                        progress_bar.update(config.batch_size)
                    train_writer.add_summary(summary, epoch * len(training_set) + k)
                    if not epoch % config.save_model_every:
                        model_path = './builds/' + timestamp
                        saver.save(sess, model_path, global_step=epoch)
            if config.debug:
                progress_bar.close()
            training_set.shuffle_lines()
            if not epoch % config.save_model_every:
                model_path = './builds/' + timestamp + '/model'
                saver.save(sess, model_path, global_step=epoch)
