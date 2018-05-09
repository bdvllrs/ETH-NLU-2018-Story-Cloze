import os
import tensorflow as tf
import time
from tqdm import tqdm
from models import Scheduler, scheduler, scheduler_preprocess, scheduler_get_labels, scheduler_optimize
from utils import load_embedding
from Dataloader import Dataloader

vocab_size = 20000
embedding_size = 100
hidden_size = 100
batch_size = 512
max_size = 50
num_rnns = 2
learning_rate = 0.1
n_epochs = 10
save_model_every = 10  # every epoch

embeddingpath = os.path.abspath(os.path.join(os.path.curdir, './wordembeddings.word2vec'))

training_set = Dataloader('data/train_stories.csv')
training_set.set_preprocess_fn(scheduler_preprocess)
training_set.set_special_tokens(['<pad>', '<unk>'])
# training_set.compute_vocab()
# training_set.save_vocab('./default.voc')
training_set.load_vocab('./default.voc', vocab_size)
# print(training_set.get(2, batch_size, random=True))

scheduler_model = Scheduler(batch_size, vocab_size, embedding_size, hidden_size)
output = scheduler_model()
scheduler_model.optimize(learning_rate)
# word_embeddings, output = scheduler(batch_size, vocab_size, embedding_size, hidden_size)
# opt, mse = scheduler_optimize(output, learning_rate, batch_size)

tf.summary.scalar("cost", scheduler_model.mse)

with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=2,
                                      intra_op_parallelism_threads=2)) as sess:
    timestamp = str(int(time.time()))
    writer = tf.summary.FileWriter('./logs/' + timestamp, sess.graph)
    sess.run(tf.global_variables_initializer())

    # Load word2vec pretrained embeddings
    # load_embedding(sess, training_set.word_to_index, word_embeddings, embeddingpath, embedding_size, vocab_size)

    computed_cross_entropy = 0
    for epoch in range(n_epochs):
        for k in tqdm(range(0, len(training_set), batch_size)):
            if k + batch_size < len(training_set):
                summary_op = tf.summary.merge_all()

                batch = training_set.get(k, batch_size, random=True)
                shuffled_batch, labels = scheduler_get_labels(batch)
                probabilities, _, computed_mse, summary = sess.run(
                    ['scheduler/order_probability:0', 'scheduler/optimize/optimizer',
                     'scheduler/optimize/mse:0', summary_op],
                    {'scheduler/x:0': shuffled_batch,
                     'scheduler/optimize/label:0': labels})
                writer.add_summary(summary, epoch * len(training_set) + k)
        training_set.shuffle_lines()
        if not epoch % save_model_every:
            tf.saved_model.simple_save(sess, './builds/' + timestamp + '/epoch-' + str(epoch),
                                       {'scheduler/x:0': scheduler_model.x,
                                        'scheduler/optimize/label:0': scheduler_model.labels},
                                       {'scheduler/order_probability:0': scheduler_model.probabilities,
                                        'scheduler/optimize/mse:0': scheduler_model.mse})
