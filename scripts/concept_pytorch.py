import datetime

from utils import Dataloader
from scripts import DefaultScript
import numpy as np
from torch.autograd import Variable
import torch
import time
from utils.Trainer import Seq2SeqTrainer
USE_CUDA = torch.cuda.is_available()
import tensorflow as tf
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
writer = tf.summary.FileWriter('./logs/' + timestamp + '-concept-fb/')
class Script(DefaultScript):
    slug = 'concept_fb'

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
        plot_losses_train = []
        # plot_losses_train_adv=[]
        plot_losses_train_cross = []
        plot_losses_train_auto = []
        plot_accurracies_avg = []
        plot_accurracies_avg_val = []
        start = time.time()
        Seq2SEq_main_model = Seq2SeqTrainer(self.config.hidden_size, self.config.embedding_size,
                                            self.config.n_layers, self.config.batch_size,
                                            self.config.attention_bolean, dropout=0.5,
                                            learning_rate=0.0003,
                                            plot_every=20, print_every=100, evaluate_every=1000)
        plot_loss_total = 0
        plot_loss_total_auto = 0
        plot_loss_total_cross = 0
        compteur_val = 0
        while epoch < self.config.n_epochs:
            print("Epoch:", epoch)
            epoch += 1
            for phase in ['train', 'test']:
                print(phase)
                if phase == 'train':
                    for num_1, batch in enumerate(generator_training):
                        print(num_1)
                        main_loss_total, loss_auto_debut, loss_auto_fin, loss_cross_debut, loss_cross_fin = Seq2SEq_main_model.train_all(
                            batch)

                        accuracy_summary = tf.Summary()
                        accuracy_summary.value.add(tag='train_loss_main', simple_value=main_loss_total)
                        accuracy_summary.value.add(tag='train_loss_auto', simple_value=loss_auto_debut+loss_auto_fin)
                        accuracy_summary.value.add(tag='train_loss_cross', simple_value=loss_cross_debut+loss_cross_fin)
                        writer.add_summary(accuracy_summary, num_1)
                        plot_loss_total += main_loss_total
                        plot_loss_total_auto += loss_auto_debut + loss_auto_fin
                        plot_loss_total_cross += loss_cross_debut + loss_cross_fin
                        if num_1 % self.config.plot_every == self.config.plot_every - 1:
                            plot_loss_avg = plot_loss_total / self.config.plot_every
                            plot_loss_auto_avg = plot_loss_total_auto / self.config.plot_every
                            plot_loss_cross_avg = plot_loss_total_cross / self.config.plot_every
                            plot_losses_train.append(plot_loss_avg)
                            # plot_losses_train_adv.append(plot_loss_adv_avg)
                            plot_losses_train_auto.append(plot_loss_auto_avg)
                            plot_losses_train_cross.append(plot_loss_cross_avg)
                            #np.save('./builds/main_loss', np.array(plot_losses_train))
                            #np.save('./builds/adv_loss', np.array(plot_losses_train_adv))
                            #np.save('./builds/auto_loss', np.array(plot_losses_train_auto))
                            #np.save('./builds/cross_loss', np.array(plot_losses_train_cross))
                            print_summary = '%s (%d %d%%) %.4f %.4f %.4f' % (
                                Seq2SEq_main_model.time_since(start, (num_1 + 1) / (90000 / 32)), (num_1 + 1),
                                (num_1 + 1) / (90000 / 32) * 100,
                                plot_loss_avg, plot_loss_auto_avg, plot_loss_cross_avg)
                            print(print_summary)
                            plot_loss_total = 0
                            plot_loss_total_auto = 0
                            plot_loss_total_cross = 0
                            compteur_val += 1
                            if compteur_val == 3:
                                compteur_val = 0
                                correct = 0
                                correctfin = 0
                                correctdebut = 0
                                dcorrect = 0
                                dcorrectfin = 0
                                dcorrectdebut = 0
                                total = 0
                                for num, batch in enumerate(generator_dev):
                                    if num < 21:
                                        all_histoire_debut_embedding = Variable(torch.FloatTensor(batch[0])).transpose(0, 1)
                                        all_histoire_fin_embedding1 = Variable(torch.FloatTensor(batch[1])).transpose(0, 1)
                                        all_histoire_fin_embedding2 = Variable(torch.FloatTensor(batch[2])).transpose(0, 1)
                                        if USE_CUDA:
                                            all_histoire_debut_embedding=all_histoire_debut_embedding.cuda()
                                            all_histoire_fin_embedding1=all_histoire_fin_embedding1.cuda()
                                            all_histoire_fin_embedding2=all_histoire_fin_embedding2.cuda()
                                        labels = Variable(torch.LongTensor(batch[3]))
                                        end = Seq2SEq_main_model.evaluate(Seq2SEq_main_model.encoder_source,
                                                                          Seq2SEq_main_model.decoder_target,
                                                                          all_histoire_debut_embedding,
                                                                          Seq2SEq_main_model.input_length_debut)
                                        debut1 = Seq2SEq_main_model.evaluate(Seq2SEq_main_model.encoder_source,
                                                                             Seq2SEq_main_model.decoder_target,
                                                                             all_histoire_fin_embedding1,
                                                                             Seq2SEq_main_model.input_length_fin)
                                        debut2 = Seq2SEq_main_model.evaluate(Seq2SEq_main_model.encoder_source,
                                                                             Seq2SEq_main_model.decoder_target,
                                                                             all_histoire_fin_embedding2,
                                                                             Seq2SEq_main_model.input_length_fin)
                                        preds, predfin, preddebut, preds_dist, predfin_dis, preddebut_dis = self.get_predict(
                                            end, debut1, debut2, all_histoire_debut_embedding.transpose(0, 1),
                                            all_histoire_fin_embedding1.transpose(0, 1),
                                            all_histoire_fin_embedding2.transpose(0, 1))


                                        preds = preds.cpu().long()
                                        predfin = predfin.cpu().long()
                                        preddebut = preddebut.cpu().long()

                                        correct += (preds == labels).sum().item()
                                        correctfin += (predfin == labels).sum().item()
                                        correctdebut += (preddebut == labels).sum().item()

                                        preds_dist = preds_dist.cpu().long()
                                        predfin_dis = predfin_dis.cpu().long()
                                        preddebut_dis = preddebut_dis.cpu().long()

                                        dcorrect += (preds_dist == labels).sum().item()
                                        dcorrectfin += (predfin_dis == labels).sum().item()
                                        dcorrectdebut += (preddebut_dis == labels).sum().item()

                                        print(correct/ total,correctfin/ total,correctdebut/ total)
                                        print(dcorrect/ total, dcorrectfin/ total, dcorrectdebut/ total)

                                        total += self.config.batch_size
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
                                            plot_accurracies_avg_val.append(plot_acc_avg)
                                            if plot_acc_avg > max_acc:
                                                torch.save(Seq2SEq_main_model.encoder_source.state_dict(),
                                                           './builds/encoder_source_best.pth')
                                                torch.save(Seq2SEq_main_model.encoder_target.state_dict(),
                                                           './builds/encoder_target_best.pth')
                                                torch.save(Seq2SEq_main_model.decoder_source.state_dict(),
                                                           './builds/decoder_source_best.pth')
                                                torch.save(Seq2SEq_main_model.decoder_target.state_dict(),
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
                                        break
                else:
                    print(phase)
                    correct = 0
                    correctfin=0
                    correctdebut=0
                    dcorrect = 0
                    dcorrectfin = 0
                    dcorrectdebut = 0
                    total = 0
                    for num, batch in enumerate(generator_dev):
                        all_histoire_debut_embedding = Variable(torch.FloatTensor(batch[0])).transpose(0, 1)
                        all_histoire_fin_embedding1 = Variable(torch.FloatTensor(batch[1])).transpose(0, 1)
                        all_histoire_fin_embedding2 = Variable(torch.FloatTensor(batch[2])).transpose(0, 1)
                        if USE_CUDA:
                            all_histoire_debut_embedding = all_histoire_debut_embedding.cuda()
                            all_histoire_fin_embedding1 = all_histoire_fin_embedding1.cuda()
                            all_histoire_fin_embedding2 = all_histoire_fin_embedding2.cuda()
                        labels = Variable(torch.LongTensor(batch[3]))
                        end = Seq2SEq_main_model.evaluate(Seq2SEq_main_model.encoder_source,
                                                          Seq2SEq_main_model.decoder_target,
                                                          all_histoire_debut_embedding,
                                                          Seq2SEq_main_model.input_length_debut)
                        debut1 = Seq2SEq_main_model.evaluate(Seq2SEq_main_model.encoder_source,
                                                             Seq2SEq_main_model.decoder_target,
                                                             all_histoire_fin_embedding1,
                                                             Seq2SEq_main_model.input_length_fin)
                        debut2 = Seq2SEq_main_model.evaluate(Seq2SEq_main_model.encoder_source,
                                                             Seq2SEq_main_model.decoder_target,
                                                             all_histoire_fin_embedding2,
                                                             Seq2SEq_main_model.input_length_fin)
                        preds, predfin,preddebut,preds_dist, predfin_dis,preddebut_dis = self.get_predict(end, debut1, debut2, all_histoire_debut_embedding.transpose(0, 1),
                                                 all_histoire_fin_embedding1.transpose(0, 1),
                                                 all_histoire_fin_embedding2.transpose(0, 1))

                        preds = preds.cpu().long()
                        predfin=predfin.cpu().long()
                        preddebut=preddebut.cpu().long()

                        correct += (preds == labels).sum().item()
                        correctfin += (predfin == labels).sum().item()
                        correctdebut += (preddebut == labels).sum().item()

                        preds_dist = preds_dist.cpu().long()
                        predfin_dis = predfin_dis.cpu().long()
                        preddebut_dis = preddebut_dis.cpu().long()

                        dcorrect += (preds_dist == labels).sum().item()
                        dcorrectfin += (predfin_dis == labels).sum().item()
                        dcorrectdebut += (preddebut_dis == labels).sum().item()

                        total += self.config.batch_size
                        accuracy_summary = tf.Summary()
                        accuracy_summary.value.add(tag='test_accuracy',
                                                   simple_value=(correct / total))
                        accuracy_summary.value.add(tag='test_accuracy_fin',
                                                   simple_value=(correctfin / total))
                        accuracy_summary.value.add(tag='test_accuracy_debut',
                                                   simple_value=(correctdebut / total))
                        accuracy_summary.value.add(tag='test_accuracy_dist',
                                                   simple_value=(dcorrect / total))
                        accuracy_summary.value.add(tag='test_accuracy_fin_dist',
                                                   simple_value=(dcorrectfin / total))
                        accuracy_summary.value.add(tag='test_accuracy_debut_dist',
                                                   simple_value=(dcorrectdebut / total))
                        writer.add_summary(accuracy_summary, num - 1)
                        if num % self.config.plot_every_test == self.config.plot_every_test - 1:
                            plot_acc_avg = correct / total
                            plot_accurracies_avg.append(plot_acc_avg)
                            #np.save('./builds/accuracy_test', np.array(plot_accurracies_avg))
                            correct = 0
                            correctfin = 0
                            correctdebut = 0
                            dcorrect = 0
                            dcorrectfin = 0
                            dcorrectdebut = 0
                            total = 0

                print('SAVE MODEL END EPOCH')
                torch.save(Seq2SEq_main_model.encoder_source.state_dict(), './builds/encoder_source_epoch' + str(epoch) + '.pth')
                torch.save(Seq2SEq_main_model.encoder_target.state_dict(), './builds/encoder_target_epoch' + str(epoch) + '.pth')
                torch.save(Seq2SEq_main_model.decoder_source.state_dict(), './builds/decoder_source_epoch' + str(epoch) + '.pth')
                torch.save(Seq2SEq_main_model.decoder_target.state_dict(), './builds/decoder_target_epoch' + str(epoch) + '.pth')

    def get_predict(self, end, debut1, debut2, all_histoire_debut_embedding, all_histoire_fin_embedding1,
                    all_histoire_fin_embedding2):
        # Todo : predict selon end, selon debur, selon les 2 (dernier fait ici)
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
                b[0],
                b[1],
                b[2],
                b[3]])
            histoire_noise_debut = self.add_noise(histoire_debut)
            histoire_embedding_debut = self.infersent(histoire_debut)
            histoire_embedding_noise_debut = self.infersent(histoire_noise_debut)
            noise_debut = histoire_embedding_noise_debut - histoire_embedding_debut
            histoire_fin = np.array([
                b[4]])
            histoire_noise_fin = self.add_noise(histoire_fin)
            histoire_embedding_fin = self.infersent(histoire_fin)
            histoire_embedding_noise_fin = self.infersent(histoire_noise_fin)
            noise_fin = histoire_embedding_noise_fin - histoire_embedding_fin
            all_histoire_debut_embedding.append(histoire_embedding_debut)
            all_histoire_fin_embedding.append(histoire_embedding_fin)
            all_histoire_noise_debut.append(histoire_embedding_noise_debut)
            all_histoire_noise_fin.append(histoire_embedding_noise_fin)
            all_noise_debut.append(noise_debut)
            all_noise_fin.append(noise_fin)
        return [np.array(all_histoire_debut_embedding), np.array(all_histoire_fin_embedding),
                np.array(all_histoire_noise_debut), np.array(all_histoire_noise_fin),
                np.array(all_noise_debut), np.array(all_noise_fin)]

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
