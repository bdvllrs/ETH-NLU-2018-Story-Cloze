import datetime
from utils import Dataloader
from scripts import DefaultScript
import numpy as np
from torch.autograd import Variable
import torch
import time
import math
from utils import Discriminator
USE_CUDA = torch.cuda.is_available()
import tensorflow as tf

timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
writer = tf.summary.FileWriter('./logs/' + timestamp + '-concept-fb/')
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

class Script(DefaultScript):
    slug = 'aligment_v2'

    def train(self):
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
        plot_loss_total_auto = 0
        plot_loss_total_cross = 0
        compteur_val = 0
        while epoch < self.config.n_epochs:
            print("Epoch:", epoch)
            epoch += 1
            for num_1, batch in enumerate(generator_training):
                print(num_1)
                all_histoire_debut_embedding = Variable(torch.FloatTensor(batch[0])).cuda()
                all_histoire_fin_embedding = Variable(torch.FloatTensor(batch[1])).cuda()
                all_histoire_debut_noise =Variable(torch.FloatTensor(batch[2])).cuda()
                all_histoire_fin_noise = Variable(torch.FloatTensor(batch[3])).cuda()
                #Model
                encoder_optimizer_source.zero_grad()
                decoder_optimizer_source.zero_grad()
                encoder_optimizer_target.zero_grad()
                decoder_optimizer_target.zero_grad()
                discriminator_optmizer.zero_grad()
                encoder_src.train(True)
                decoder_src.train(True)
                encoder_tgt.train(True)
                decoder_tgt.train(True)
                discriminator.train(True)
                #autoencoder
                print("autoencoder")
                #1
                z_src_autoencoder=encoder_src(all_histoire_debut_noise)
                pred_discriminator_src=discriminator.forward(z_src_autoencoder)
                pred_discriminator_src = pred_discriminator_src.view(-1)
                adv_loss1 = criterion_adver(pred_discriminator_src, target_adver_src)
                out_src_auto = decoder_src(z_src_autoencoder)
                loss1=torch.nn.functional.cosine_embedding_loss(out_src_auto.transpose(0,2),all_histoire_debut_noise.transpose(0,2),target_adver_tgt)
                #2
                z_tgt_autoencoder = encoder_tgt(all_histoire_fin_noise)
                pred_discriminator_tgt = discriminator.forward(z_tgt_autoencoder)
                pred_discriminator_tgt = pred_discriminator_tgt.view(-1)
                adv_loss2 = criterion_adver(pred_discriminator_tgt, target_adver_tgt)
                out_tgt_auto = decoder_tgt(z_tgt_autoencoder)
                loss2 = torch.nn.functional.cosine_embedding_loss(out_tgt_auto.transpose(0, 2),
                                                                  all_histoire_fin_embedding.transpose(0, 2),
                                                                  target_adver_tgt)
                print("crossentropy")
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
                #evaluate1
                z_src_cross=encoder_src(y_src_eval)
                pred_fin = decoder_tgt(z_src_cross)
                pred_discriminator_src = discriminator.forward(z_src_cross)
                pred_discriminator_src = pred_discriminator_src.view(-1)
                adv_loss3 = criterion_adver(pred_discriminator_src, target_adver_src)
                loss3 = torch.nn.functional.cosine_embedding_loss(pred_fin.transpose(0, 2),
                                                                  all_histoire_fin_embedding.transpose(0, 2),
                                                                  target_adver_tgt)
                # evaluate2
                z_tgt_cross = encoder_tgt(y_tgt_eval)
                pred_debut = decoder_src(z_tgt_cross)
                pred_discriminator_tgt = discriminator.forward(z_tgt_cross)
                pred_discriminator_tgt = pred_discriminator_tgt.view(-1)
                adv_loss4 = criterion_adver(pred_discriminator_tgt, target_adver_tgt)
                loss4 = torch.nn.functional.cosine_embedding_loss(pred_debut.transpose(0, 2),
                                                                  all_histoire_debut_embedding.transpose(0, 2),
                                                                  target_adver_tgt)
                total_loss=loss1+loss2+loss3+loss4+adv_loss1+adv_loss2+adv_loss3+adv_loss4
                #upgrade
                total_loss.backward()
                encoder_optimizer_source.step()
                decoder_optimizer_source.step()
                encoder_optimizer_target.step()
                decoder_optimizer_target.step()
                discriminator_optmizer.step()
                accuracy_summary = tf.Summary()

                main_loss_total=total_loss.item()
                accuracy_summary.value.add(tag='train_loss_main', simple_value=main_loss_total)
                writer.add_summary(accuracy_summary, num_1)
                plot_loss_total += main_loss_total
                if num_1 % self.config.plot_every == self.config.plot_every - 1:
                    plot_loss_avg = plot_loss_total / self.config.plot_every
                    plot_loss_auto_avg = plot_loss_total_auto / self.config.plot_every
                    plot_loss_cross_avg = plot_loss_total_cross / self.config.plot_every
                    print_summary = '%s (%d %d%%) %.4f %.4f %.4f' % (
                    self.time_since(start, (num_1 + 1) / (90000 / 32)), (num_1 + 1),
                        (num_1 + 1) / (90000 / 32) * 100,
                        plot_loss_avg, plot_loss_auto_avg, plot_loss_cross_avg)
                    print(print_summary)
                    plot_loss_total = 0
                    plot_loss_total_auto = 0
                    plot_loss_total_cross = 0
                    compteur_val += 1
                    if compteur_val == 1:
                        compteur_val = 0
                        correct = 0
                        correctfin = 0
                        correctdebut = 0
                        dcorrect = 0
                        dcorrectfin = 0
                        dcorrectdebut = 0
                        total = 0
                        for num, batch in enumerate(generator_dev):
                            encoder_src.train(False)
                            decoder_src.train(False)
                            encoder_tgt.train(False)
                            decoder_tgt.train(False)
                            if num < 21:
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
                                sim1 = torch.nn.functional.cosine_embedding_loss(end.transpose(0, 2),
                                                                                  all_histoire_debut_embedding.transpose(
                                                                                      0, 2),
                                                                                  all_histoire_fin_embedding1,reduce=False)
                                sim2 = torch.nn.functional.cosine_embedding_loss(end.transpose(0, 2),
                                                                                  all_histoire_debut_embedding.transpose(
                                                                                      0, 2),
                                                                                  all_histoire_fin_embedding2)
                                preds=pred1<pred2
                                preds_sim=sim1<sim2
                                correct += (preds == labels).sum().item()
                                correctdebut += (preds_sim == labels).sum().item()
                                total += self.config.batch_size
                                print("Accuracy ")
                                print(correct/ total,correctdebut/ total)



                                accuracy_summary = tf.Summary()
                                accuracy_summary.value.add(tag='val_accuracy',
                                                           simple_value=(correct / total))
                                accuracy_summary.value.add(tag='val_accuracy_fin',
                                                           simple_value=(correctfin / total))
                                accuracy_summary.value.add(tag='val_accuracy_debut',
                                                           simple_value=(correctdebut / total))
                                accuracy_summary.value.add(tag='val_accuracy_dist',
                                                           simple_value=(dcorrect / total))
                                accuracy_summary.value.add(tag='val_accuracy_fin_dist',
                                                           simple_value=(dcorrectfin / total))
                                accuracy_summary.value.add(tag='val_accuracy_debut_dist',
                                                           simple_value=(dcorrectdebut / total))
                                writer.add_summary(accuracy_summary, num + num_1 - 1)
                                if num % self.config.plot_every_test == self.config.plot_every_test - 1:
                                    plot_acc_avg = correct / total
                                    if plot_acc_avg > max_acc:
                                        torch.save(encoder_src.state_dict(),
                                                   './builds/encoder_source_best.pth')
                                        torch.save(encoder_tgt.state_dict(),
                                                   './builds/encoder_target_best.pth')
                                        torch.save(decoder_src.decoder_source.state_dict(),
                                                   './builds/decoder_source_best.pth')
                                        torch.save( decoder_tgt.state_dict(),
                                                   './builds/decoder_target_best.pth')
                                        max_acc = plot_acc_avg
                                        print('SAVE MODEL FOR ACCURACY : ' + str(plot_acc_avg))
                                    correct = 0
                                    correctfin=0
                                    correctdebut=0
                                    dcorrect = 0
                                    dcorrectfin = 0
                                    dcorrectdebut = 0
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

    def get_predict(self, end, debut1, debut2, all_histoire_debut_embedding, all_histoire_fin_embedding1,
                    all_histoire_fin_embedding2):
        """
        :param end:
        :param debut1:
        :param debut2:
        :param all_histoire_debut_embedding:
        :param all_histoire_fin_embedding1:
        :param all_histoire_fin_embedding2:
        :return:
        """
        semblable_fin1 = []
        semblable_fin2 = []
        semblable_debut1 = []
        semblable_debut2 = []
        debut1 = debut1.cpu()
        debut2 = debut2.cpu()
        all_histoire_debut_embedding = all_histoire_debut_embedding.cpu()
        end = end.cpu()
        all_histoire_fin_embedding1 = all_histoire_fin_embedding1.cpu()
        all_histoire_fin_embedding2 = all_histoire_fin_embedding2.cpu()
        for num_batch, batch in enumerate(end):
            for num_sent, sent in enumerate(batch):
                produit = torch.dot(sent.float(), all_histoire_fin_embedding1[num_batch][num_sent].float()) / (
                            torch.norm(sent) * torch.norm(
                        all_histoire_fin_embedding1[num_batch][num_sent]))
                semblable_fin1.append(produit.cpu().data[0])
                produit = torch.dot(sent.float(), all_histoire_fin_embedding2[num_batch][num_sent].float()) / (
                        torch.norm(sent.float()) * torch.norm(all_histoire_fin_embedding2[num_batch][num_sent].float()))
                semblable_fin2.append(produit.cpu().data[0])
                produit = torch.dot(debut1[num_batch][num_sent].float(),
                                    all_histoire_debut_embedding[num_batch][num_sent].float()) / (
                                  torch.norm(debut1[num_batch][num_sent].float()) * torch.norm(
                              all_histoire_debut_embedding[num_batch][num_sent].float()))
                semblable_debut1.append(produit.cpu().data[0])
                produit = torch.dot(debut2[num_batch][num_sent].float(),
                                    all_histoire_debut_embedding[num_batch][num_sent].float()) / (
                                  torch.norm(debut2[num_batch][num_sent].float()) * torch.norm(
                                  all_histoire_debut_embedding[num_batch][num_sent].float()))
                semblable_debut2.append(produit.cpu().data[0])
        p1 = Variable(torch.FloatTensor((np.array(semblable_fin1) + np.array(semblable_debut1)) / 2))
        p2 = Variable(torch.FloatTensor((np.array(semblable_fin2) + np.array(semblable_debut2)) / 2))
        p = torch.stack((p1, p2))
        pf = torch.stack((torch.FloatTensor((np.array(semblable_fin1))), torch.FloatTensor((np.array(semblable_fin2)))))
        pd = torch.stack((torch.FloatTensor((np.array(semblable_debut1))), torch.FloatTensor((np.array(semblable_debut2)))))
        (_, pred) = torch.max(p, 0)
        (_, predfin) = torch.max(pf, 0)
        (_, preddebut) = torch.max(pd, 0)
        distancee1=end-all_histoire_fin_embedding1
        distancee2=end-all_histoire_fin_embedding2
        distanced1=debut1-all_histoire_debut_embedding
        distanced2=debut2-all_histoire_debut_embedding
        df0=[]
        df00=[]
        df000=[]
        for num_batch, batch in enumerate(distancee1):
            p1=torch.norm(batch)
            p2=torch.norm(distancee2[num_batch])
            p = torch.stack((p1, p2))
            (_, predfin_distance) = torch.min(p, 0)
            df0.append(predfin_distance.item())
            p1 = torch.norm(distanced1[num_batch])
            p2 = torch.norm(distanced2[num_batch])
            p = torch.stack((p1, p2))
            (_, preddebut_distance) = torch.min(p, 0)
            df00.append(preddebut_distance.item())
            p1 = torch.norm(distanced1[num_batch]+batch)
            p2 = torch.norm(distanced2[num_batch]+distancee2[num_batch])
            p = torch.stack((p1, p2))
            (_, pred_distance) = torch.min(p, 0)
            df000.append(pred_distance.item())

        return (pred,predfin,preddebut,Variable(torch.LongTensor(df0)),Variable(torch.LongTensor(df00)),Variable(torch.LongTensor(df000)))

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
        all_noise_debut = []
        all_noise_fin = []
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
                b[0],
                b[1],
                b[2],
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

