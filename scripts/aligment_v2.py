import datetime
from utils import Dataloader
from scripts import DefaultScript
import numpy as np
from torch.autograd import Variable
import torch
import time
import math
import random
from utils import Discriminator
USE_CUDA = torch.cuda.is_available()
import tensorflow as tf

from torch import optim
import torch.nn as nn
import torch.nn.functional as F

class Script(DefaultScript):
    slug = 'aligment_v2'

    def train(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer = tf.summary.FileWriter('./logs/' + timestamp + '-concept-fb/')
        output_fn = OutputFN(self.config.GLOVE_PATH, self.config.model_path)
        train_set = Dataloader(self.config, 'data/train_stories.csv')
        test_set = Dataloader(self.config, 'data/test_stories.csv', testing_data=True)
        train_set.set_special_tokens(["<unk>"])
        test_set.set_special_tokens(["<unk>"])
        train_set.load_dataset('data/train.bin')
        train_set.load_vocab('./data/default.voc', self.config.vocab_size)
        test_set.load_dataset('data/test.bin')
        test_set.load_vocab('./data/default.voc', self.config.vocab_size)
        test_set.set_output_fn(output_fn.output_fn_test)
        train_set.set_output_fn(output_fn)
        generator_training = train_set.get_batch(self.config.batch_size, 1)
        generator_dev = test_set.get_batch(self.config.batch_size, 1)
        epoch = 0
        max_acc = 0
        start = time.time()
        encoder_src=Encoder_src().cuda()
        decoder_src = Decoder_src().cuda()
        encoder_tgt = Encoder_tgt().cuda()
        decoder_tgt = Decoder_tgt().cuda()
        discriminator = Discriminator().cuda()
        encoder_optimizer_source = optim.Adam(encoder_src.parameters(), lr=self.config.learning_rate)
        decoder_optimizer_source = optim.Adam(decoder_src.parameters(),
                                                   lr=self.config.learning_rate)
        encoder_optimizer_target = optim.Adam(encoder_tgt.parameters(), lr=self.config.learning_rate)
        decoder_optimizer_target = optim.Adam(decoder_tgt.parameters(),
                                                   lr=self.config.learning_rate)
        discriminator_optmizer = optim.RMSprop(discriminator.parameters(), lr=self.config.learning_rate_discriminator)
        criterion_adver = nn.BCELoss()
        target_adver_src = Variable(torch.zeros(self.config.batch_size)).cuda()
        target_adver_tgt = Variable(torch.ones(self.config.batch_size)).cuda()
        plot_loss_total = 0
        plot_loss_total_adv = 0


        compteur=0
        compteur_val = 0
        while epoch < self.config.n_epochs:
            print("Epoch:", epoch)
            epoch += 1
            for num_1, batch in enumerate(generator_training):
                all_histoire_debut_embedding = Variable(torch.FloatTensor(batch[0])).cuda()
                all_histoire_fin_embedding = Variable(torch.FloatTensor(batch[1])).cuda()
                all_histoire_debut_noise =Variable(torch.FloatTensor(batch[2])).cuda()
                all_histoire_fin_noise = Variable(torch.FloatTensor(batch[3])).cuda()
                if num_1%2==0:
                    encoder_optimizer_source.zero_grad()
                    decoder_optimizer_source.zero_grad()
                    encoder_optimizer_target.zero_grad()
                    decoder_optimizer_target.zero_grad()
                    encoder_src.train(True)
                    decoder_src.train(True)
                    encoder_tgt.train(True)
                    decoder_tgt.train(True)
                    discriminator.train(False)
                    z_src_autoencoder=encoder_src(all_histoire_debut_noise)
                    out_src_auto = decoder_src(z_src_autoencoder)
                    loss1=torch.nn.functional.cosine_embedding_loss(out_src_auto.transpose(0,2).transpose(0,1),all_histoire_debut_noise.transpose(0,2).transpose(0,1),target_adver_tgt)
                    z_tgt_autoencoder = encoder_tgt(all_histoire_fin_noise)
                    out_tgt_auto = decoder_tgt(z_tgt_autoencoder)
                    loss2 = torch.nn.functional.cosine_embedding_loss(out_tgt_auto.transpose(0, 2).transpose(0,1),
                                                                      all_histoire_fin_embedding.transpose(0, 2).transpose(0,1),
                                                                      target_adver_tgt)
                    if epoch == 1:
                        y_src_eval= all_histoire_debut_noise
                        y_tgt_eval = all_histoire_fin_noise
                    else:
                        encoder_src.train(False)
                        decoder_src.train(False)
                        encoder_tgt.train(False)
                        decoder_tgt.train(False)
                        y_tgt_eval = decoder_tgt(encoder_src(all_histoire_debut_embedding))
                        y_src_eval = decoder_src(encoder_tgt(all_histoire_fin_embedding))
                        encoder_src.train(True)
                        decoder_src.train(True)
                        encoder_tgt.train(True)
                        decoder_tgt.train(True)

                    z_src_cross=encoder_src(y_src_eval)
                    pred_fin = decoder_tgt(z_src_cross)
                    loss3 = torch.nn.functional.cosine_embedding_loss(pred_fin.transpose(0, 2).transpose(0,1),
                                                                      all_histoire_fin_embedding.transpose(0, 2).transpose(0,1),
                                                                      target_adver_tgt)
                    # evaluate2
                    z_tgt_cross = encoder_tgt(y_tgt_eval)
                    pred_debut = decoder_src(z_tgt_cross)
                    loss4 = torch.nn.functional.cosine_embedding_loss(pred_debut.transpose(0, 2).transpose(0,1),
                                                                      all_histoire_debut_embedding.transpose(0, 2).transpose(0,1),
                                                                      target_adver_tgt)
                    total_loss=loss1+loss2+loss3+loss4

                    total_loss.backward()
                    encoder_optimizer_source.step()
                    decoder_optimizer_source.step()
                    encoder_optimizer_target.step()
                    decoder_optimizer_target.step()
                    accuracy_summary = tf.Summary()
                    main_loss_total=total_loss.item()
                    accuracy_summary.value.add(tag='train_loss_main', simple_value=main_loss_total)
                    writer.add_summary(accuracy_summary, num_1)
                    plot_loss_total += main_loss_total
                else:
                    #REDEFINIR lINPUT du DATASET
                    new_X=[]
                    new_Y_src=[]
                    new_Y_tgt=[]
                    for num_3,batch_3 in enumerate(all_histoire_debut_embedding):

                        if random.random()>0.5:

                            new_X.append(batch_3.cpu().numpy())
                            new_Y_src.append(1)
                            new_Y_tgt.append(0)
                        else:
                            new_X.append(all_histoire_fin_embedding[num_3].cpu().numpy())
                            new_Y_src.append(0)
                            new_Y_tgt.append(1)
                    all_histoire_debut_noise=Variable(torch.FloatTensor(np.array(new_X))).cuda()
                    target_adver_src=Variable(torch.FloatTensor(np.array(new_Y_src))).cuda()
                    target_adver_tgt=Variable(torch.FloatTensor(np.array(new_Y_tgt))).cuda()
                    discriminator_optmizer.zero_grad()
                    discriminator.train(True)
                    encoder_src.train(False)
                    decoder_src.train(False)
                    encoder_tgt.train(False)
                    decoder_tgt.train(False)
                    z_src_autoencoder=encoder_src(all_histoire_debut_noise)
                    pred_discriminator_src=discriminator.forward(z_src_autoencoder)
                    pred_discriminator_src = pred_discriminator_src.view(-1)
                    adv_loss1 = criterion_adver(pred_discriminator_src, target_adver_src)
                    z_tgt_autoencoder = encoder_tgt(all_histoire_debut_noise)
                    pred_discriminator_tgt = discriminator.forward(z_tgt_autoencoder)
                    pred_discriminator_tgt = pred_discriminator_tgt.view(-1)
                    adv_loss2 = criterion_adver(pred_discriminator_tgt, target_adver_tgt)
                    if epoch == 1:
                        y_src_eval= all_histoire_debut_noise
                        y_tgt_eval = all_histoire_fin_noise
                    else:
                        encoder_src.train(False)
                        decoder_src.train(False)
                        encoder_tgt.train(False)
                        decoder_tgt.train(False)
                        y_tgt_eval = decoder_tgt(encoder_src(all_histoire_debut_embedding))
                        y_src_eval = decoder_src(encoder_tgt(all_histoire_debut_embedding))
                        encoder_src.train(True)
                        decoder_src.train(True)
                        encoder_tgt.train(True)
                        decoder_tgt.train(True)
                        #evaluate1
                    z_src_cross=encoder_src(y_src_eval)
                    pred_discriminator_src = discriminator.forward(z_src_cross)
                    pred_discriminator_src = pred_discriminator_src.view(-1)
                    adv_loss3 = criterion_adver(pred_discriminator_src, target_adver_src)
                    # evaluate2
                    z_tgt_cross = encoder_tgt(y_tgt_eval)
                    pred_discriminator_tgt = discriminator.forward(z_tgt_cross)
                    pred_discriminator_tgt = pred_discriminator_tgt.view(-1)
                    adv_loss4 = criterion_adver(pred_discriminator_tgt, target_adver_tgt)
                    total_loss_adv=adv_loss1+adv_loss2+adv_loss3+adv_loss4
                    total_loss_adv.backward()
                    discriminator_optmizer.step()
                    accuracy_summary = tf.Summary()
                    main_loss_total_adv=total_loss_adv.item()
                    accuracy_summary.value.add(tag='train_loss_main_adv', simple_value=main_loss_total_adv)
                    writer.add_summary(accuracy_summary, num_1)
                    plot_loss_total_adv += main_loss_total_adv
                if num_1 % self.config.plot_every == self.config.plot_every - 1:
                    plot_loss_avg = plot_loss_total / self.config.plot_every
                    plot_loss_avg_adv = plot_loss_total_adv / self.config.plot_every
                    print_summary = '%s (%d %d%%) %.4f %.4f' % (
                    self.time_since(start, (num_1 + 1) / (90000 / 32)), (num_1 + 1),
                        (num_1 + 1) / (90000 / 32) * 100,
                        plot_loss_avg,plot_loss_avg_adv)
                    print(print_summary)
                    plot_loss_total = 0
                    compteur_val += 1
                    if compteur_val == 3:
                        compteur
                        compteur_val = 0
                        correct = 0
                        correctfin = 0
                        correctdebut = 0
                        total = 0
                        for num, batch in enumerate(generator_dev):
                            compteur+=1
                            encoder_src.train(False)
                            decoder_src.train(False)
                            encoder_tgt.train(False)
                            decoder_tgt.train(False)
                            discriminator.train(False)
                            if num < 11:
                                all_histoire_debut_embedding = Variable(torch.FloatTensor(batch[0]))
                                all_histoire_fin_embedding1 = Variable(torch.FloatTensor(batch[1]))
                                all_histoire_fin_embedding2 = Variable(torch.FloatTensor(batch[2]))
                                if USE_CUDA:
                                    all_histoire_debut_embedding=all_histoire_debut_embedding.cuda()
                                    all_histoire_fin_embedding1=all_histoire_fin_embedding1.cuda()
                                    all_histoire_fin_embedding2=all_histoire_fin_embedding2.cuda()
                                labels = Variable(torch.LongTensor(batch[3]))
                                end = decoder_tgt(encoder_src(all_histoire_debut_embedding))
                                z_end1 =encoder_src(all_histoire_fin_embedding1)
                                z_end2=encoder_src(all_histoire_fin_embedding2)


                                pred1 = discriminator.forward(z_end1)
                                pred1 = pred1.view(-1)
                                pred2 = discriminator.forward(z_end2)
                                pred2 = pred2.view(-1)

                                sim1 = torch.nn.functional.cosine_embedding_loss(end.transpose(0, 2).transpose(0,1),
                                                                                  all_histoire_fin_embedding1.transpose(
                                                                                      0, 2).transpose(0,1),
                                                                                  target_adver_tgt,reduce=False)
                                sim2 = torch.nn.functional.cosine_embedding_loss(end.transpose(0, 2).transpose(0,1),
                                                                                  all_histoire_fin_embedding1.transpose(
                                                                                      0, 2).transpose(0,1),
                                                                                  target_adver_tgt,reduce=False)
                                preds=(pred1<pred2).cpu().long()
                                preds_sim=(sim1>sim2).cpu().long()

                                correct += (preds == labels).sum().item()
                                correctdebut += (preds_sim == labels).sum().item()
                                total += self.config.batch_size
                                print("Accuracy ")
                                print(correct/ total,correctdebut/ total)
                                accuracy_summary = tf.Summary()
                                accuracy_summary.value.add(tag='val_accuracy',
                                                           simple_value=(correct / total))
                                accuracy_summary.value.add(tag='val_accuracy_similitude',
                                                           simple_value=(correctfin / total))
                                writer.add_summary(accuracy_summary,compteur)
                                if num % self.config.plot_every_test == self.config.plot_every_test - 1:
                                    plot_acc_avg = correct / total
                                    if plot_acc_avg > max_acc:
                                        torch.save(encoder_src.state_dict(),
                                                   './builds/encoder_source_best.pth')
                                        torch.save(encoder_tgt.state_dict(),
                                                   './builds/encoder_target_best.pth')
                                        torch.save(decoder_src.state_dict(),
                                                   './builds/decoder_source_best.pth')
                                        torch.save( decoder_tgt.state_dict(),
                                                   './builds/decoder_target_best.pth')
                                        max_acc = plot_acc_avg
                                        print('SAVE MODEL FOR ACCURACY : ' + str(plot_acc_avg))
                                    correct = 0
                                    correctfin=0
                                    correctdebut=0
                                    total = 0
                            else:
                                print('done validation')
                                encoder_src.train(True)
                                decoder_src.train(True)
                                encoder_tgt.train(True)
                                decoder_tgt.train(True)
                                break

            print('SAVE MODEL END EPOCH')
            torch.save(encoder_src.state_dict(), './builds/encoder_source_epoch' + str(epoch) + '.pth')
            torch.save(encoder_tgt.state_dict(), './builds/encoder_target_epoch' + str(epoch) + '.pth')
            torch.save(decoder_src.state_dict(), './builds/decoder_source_epoch' + str(epoch) + '.pth')
            torch.save(decoder_tgt.state_dict(), './builds/decoder_target_epoch' + str(epoch) + '.pth')

    def as_minutes(self,s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(self,since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.as_minutes(s), self.as_minutes(rs))


class OutputFN:

    def __init__(self, GLOVE_PATH, model_path):
        self.GLOVE_PATH = GLOVE_PATH
        if USE_CUDA:
            self.model = torch.load(model_path)
        else:
            self.model=torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.set_glove_path(self.GLOVE_PATH)
        self.model.build_vocab_k_words(K=100000)

    def __call__(self, data):
        batch = np.array(data.batch)
        all_histoire_debut_embedding = []
        all_histoire_fin_embedding = []
        all_histoire_noise_debut = []
        all_histoire_noise_fin = []
        for b in batch:
            histoire_debut = np.array([
                b[3]])
            histoire_noise_debut = self.add_noise(histoire_debut)
            histoire_embedding_debut = self.infersent(histoire_debut)
            histoire_embedding_noise_debut = self.infersent(histoire_noise_debut)
            histoire_fin = np.array([
                b[4]])
            histoire_noise_fin = self.add_noise(histoire_fin)
            histoire_embedding_fin = self.infersent(histoire_fin)
            histoire_embedding_noise_fin = self.infersent(histoire_noise_fin)
            all_histoire_debut_embedding.append(histoire_embedding_debut)
            all_histoire_fin_embedding.append(histoire_embedding_fin)
            all_histoire_noise_debut.append(histoire_embedding_noise_debut)
            all_histoire_noise_fin.append(histoire_embedding_noise_fin)
        return [np.array(all_histoire_debut_embedding), np.array(all_histoire_fin_embedding),
                np.array(all_histoire_noise_debut), np.array(all_histoire_noise_fin)]

    def infersent(self, story):
        """
        :param story:
        :return:
        """
        sentences = []
        for num, sto in enumerate(story):
            sto = ' '.join(sto)
            sentences.append(sto)
        embeddings = self.model.encode(sentences, tokenize=True, verbose=True)
        return (embeddings)

    def output_fn_test(self, data):
        """
        :param data:
        :return:
        """
        batch = np.array(data.batch)
        all_histoire_debut_embedding = []
        all_histoire_fin_embedding1 = []
        all_histoire_fin_embedding2 = []
        label = []
        for b in batch:
            histoire_debut = np.array([
                b[3]])
            histoire_embedding_debut = self.infersent(histoire_debut)
            all_histoire_debut_embedding.append(histoire_embedding_debut)
            histoire_fin1 = np.array([
                b[4]])
            histoire_fin2 = np.array([
                b[5]])
            histoire_embedding_fin1 = self.infersent(histoire_fin1)
            all_histoire_fin_embedding1.append(histoire_embedding_fin1)
            histoire_embedding_fin2 = self.infersent(histoire_fin2)
            all_histoire_fin_embedding2.append(histoire_embedding_fin2)
            label.append(2 - int(b[6][0]))
        return [np.array(all_histoire_debut_embedding), np.array(all_histoire_fin_embedding1),
                np.array(all_histoire_fin_embedding2), np.array(label)]

    def add_noise(self, variable, drop_probability: float = 0.1, shuffle_max_distance: int = 3):
        """
        :param variable:np array that : [[sentence1][sentence2]]
        :param drop_probability: we drop every word in the input sentence with a probability
        :param shuffle_max_distance: we slightly shuffle the input sentence
        :return:
        """

        def perm(i):
            return i[0] + (shuffle_max_distance + 1) * np.random.random()

        liste = []
        for b in range(variable.shape[0]):
            sequence = variable[b]
            if (type(sequence) != list):
                sequence = sequence.tolist()
            sequence, reminder = sequence[:-1], sequence[-1:]
            if len(sequence) != 0:
                compteur = 0
                for num, val in enumerate(np.random.random_sample(len(sequence))):
                    if val < drop_probability:
                        sequence.pop(num - compteur)
                        compteur = compteur + 1
                sequence = [x for _, x in sorted(enumerate(sequence), key=perm)]
            sequence = np.concatenate((sequence, reminder), axis=0)
            liste.append(sequence)
        new_variable = np.array(liste)
        return new_variable






class Encoder_src(nn.Module):

    def __init__(self):
        super(Encoder_src, self).__init__()

        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class Decoder_src(nn.Module):

    def __init__(self):
        super(Decoder_src, self).__init__()

        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class Encoder_tgt(nn.Module):

    def __init__(self):
        super(Encoder_tgt, self).__init__()

        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class Decoder_tgt(nn.Module):

    def __init__(self):
        super(Decoder_tgt, self).__init__()

        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

