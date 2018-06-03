import math
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from utils.Discriminator import Discriminator
from models.Seq2Seq import EncoderRNN, DecoderStep
import numpy as np
USE_CUDA = torch.cuda.is_available()
print('On utilise le GPU, ' + str(USE_CUDA) + ' story')


class Seq2SeqTrainer(nn.Module):
    """
    Model used for the human preferences metric
    """

    def __init__(self, hidden_size, embed_size, n_layers,batch_size,
                 attention_bolean, dropout=0.5, learning_rate=0.0003,
                 plot_every=20, print_every=100, evaluate_every=1000, max_length=100,learning_rate_discriminator=0.0005):
        super(Seq2SeqTrainer, self).__init__()
        """
        :param input_size:
        :param embed_size:
        :param hidden_size:
        :param output_size:
        :param indextoword:
        :param wordtoindex
        :param num_layer_encoder:
        :param num_layer_decoder:
        :param dropout:
        :param SOS_token:
        :param EOS_token:
        :param PAD_token:
        :param batch_size:
        :param article_max_size:
        """
        # Configure models
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.embed_size = embed_size
        # Configure training/optimization
        self.learning_rate = learning_rate
        self.plot_every = plot_every
        self.print_every = print_every
        self.evaluate_every = evaluate_every
        # config type model
        self.attention_bolean = attention_bolean
        self.learning_rate_discriminator=learning_rate_discriminator
        self.input_length_debut = Variable(torch.from_numpy(np.array([4] * self.batch_size, dtype=np.int32)).long()).cuda()
        self.input_length_fin = Variable(torch.LongTensor(np.array([1] * self.batch_size, dtype=np.int32)).long()).cuda()
        # Initialize models
        self.encoder_source = EncoderRNN(4, embed_size, hidden_size, n_layers=n_layers, dropout=dropout)
        self.decoder_source = DecoderStep(4,hidden_size, embed_size, n_layers, dropout_p=dropout,
                                   attention_bol=self.attention_bolean)
        self.encoder_target = EncoderRNN( 1,embed_size, hidden_size, n_layers=n_layers,dropout=dropout)
        self.decoder_target = DecoderStep(1,hidden_size, embed_size, n_layers, dropout_p=dropout,
                                   attention_bol=self.attention_bolean)

        self.decoder_source.attn_encoder=self.decoder_target.attn_encoder

        self.discriminator = Discriminator(max_length, hidden_size, hidden_size, n_layers)
        self.encoder_optimizer_source = optim.Adam(self.encoder_source.parameters(), lr=self.learning_rate)
        self.decoder_optimizer_source = optim.Adam(self.decoder_source.parameters(),
                                              lr=self.learning_rate)
        self.encoder_optimizer_target = optim.Adam(self.encoder_target.parameters(), lr=self.learning_rate)
        self.decoder_optimizer_target = optim.Adam(self.decoder_target.parameters(),
                                              lr=self.learning_rate)
        self.discriminator_optmizer=optim.RMSprop(self.discriminator.parameters(), lr=self.learning_rate_discriminator)
        # Move models to GPU
        if USE_CUDA:
            self.encoder_source.cuda()
            self.decoder_source.cuda()
            self.encoder_target.cuda()
            self.decoder_target.cuda()
            self.discriminator.cuda()

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






    def train_step(self, encoder, decoder, input_batches, input_lengths, target_batches, target_lengths, encoder_optimizer,
                   decoder_optimizer, discriminator_optmizer, criterion_adver, target_adver):
        """
        :param encoder:
        :param decoder:
        :param input_batches:
        :param input_lengths:
        :param target_batches:
        :param target_lengths:
        :param encoder_optimizer:
        :param decoder_optimizer:
        :param discriminator_optmizer:
        :param criterion_adver:
        :param target_adver:
        :return:
        """
        # Zero gradients of both optimizers
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        discriminator_optmizer.zero_grad()
        encoder.train(True)
        decoder.train(True)
        # Run words through encoder
        encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
        #Discriminator action
        log_prob = self.discriminator.forward(encoder_outputs)
        log_prob = log_prob.view(-1)
        adv_loss = criterion_adver(log_prob, target_adver)
        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([[0] * self.batch_size]*self.embed_size))
        decoder_hidden = encoder_hidden[:self.n_layers]  # Use last (forward) hidden state from encoder
        max_target_length = int(np.amax(target_lengths.cpu().numpy()))
        all_decoder_outputs = Variable(torch.zeros(max_target_length, self.batch_size, self.decoder_source.embed_size))
        # Move new Variables to CUDA
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            all_decoder_outputs = all_decoder_outputs.cuda()
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_output, output_concat, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            all_decoder_outputs[t] = decoder_output
            decoder_input = target_batches[t].transpose(0,1)  # Next input is current target
        # Loss calculation and backpropagation
        total_loss =1- torch.cos(all_decoder_outputs.float()-target_batches.float())
        total_loss=total_loss.sum(2)/self.embed_size
        total_loss = total_loss.sum(0) / max_target_length
        total_loss = total_loss.sum(0) / self.batch_size
        total_loss.backward(retain_graph=True)
        adv_loss.backward()
        # Update parameters with optimizers
        encoder_optimizer.step()
        decoder_optimizer.step()
        discriminator_optmizer.step()
        return total_loss.item(),adv_loss.item()


    def evaluate(self,encoder,decoder, batch_input, input_lengths):
        """
        :param encoder:
        :param decoder:
        :param batch_input:
        :param input_lengths:
        :return:
        """
        input_batches=batch_input
        # Set to not-training mode to disable dropout
        encoder.train(False)
        decoder.train(False)
        # Run through encoder
        encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
        # Share embedding
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # Create starting vectors for decoder
        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([[0] * self.batch_size] * self.embed_size))
        decoder_hidden = encoder_hidden[:self.n_layers]  # Use last (forward) hidden state from encoder
        max_target_length = 5-int(np.amax(input_lengths.cpu().numpy()))
        all_decoder_outputs = Variable(torch.zeros(max_target_length, self.batch_size, self.decoder_source.embed_size))
        # Move new Variables to CUDA
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            all_decoder_outputs = all_decoder_outputs.cuda()
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_output, output_concat, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            all_decoder_outputs[t] = decoder_output
        # Set back to training mode
        encoder.train(True)
        decoder.train(True)
        return all_decoder_outputs.transpose(0, 1).contiguous()


    def train_auto_encoder(self, input, noise_input,input_lengths,target_lengths,start,encoder_optimizer_source,decoder_optimizer_source,discriminator_optmizer, criterion_adver,encoder, decoder):
        """
        :param encoder:
        :param decoder:
        :param input:
        :param input_length:
        :param output:
        :param output_length:
        :return:
        """
        if start:
            target_adver = Variable(torch.zeros(self.batch_size)).cuda()  # B
        else:
            target_adver = Variable(torch.ones(self.batch_size)).cuda()  # B
        noise_input = Variable(torch.LongTensor(noise_input)).cuda()
        input = Variable(torch.LongTensor(input)).cuda()
        loss_start1, adv_loss_start1= self.train_step(encoder, decoder, noise_input.transpose(0,1),input_lengths,
                                                                                               input.transpose(0, 1),
                                                                                               target_lengths,
                                                                                               encoder_optimizer_source,
                                                                                               decoder_optimizer_source,
                                                                                               discriminator_optmizer,
                                                                                               criterion_adver,
                                                                                               target_adver)
        return(loss_start1, adv_loss_start1 )

    def train_cross(self, input,input_length, output_length,noise, start, encoder_optimizer_source, decoder_optimizer_source,
                           discriminator_optmizer, criterion_adver, encoder, decoder):
        """
        :param input:
        :param output:
        :param all_batch:
        :param start:
        :param encoder_optimizer_source:
        :param decoder_optimizer_source:
        :param discriminator_optmizer:
        :param criterion_adver:
        :return:
        """
        if start:
            target_adver = Variable(torch.zeros(self.batch_size)).cuda()  # B
        else:
            target_adver = Variable(torch.ones(self.batch_size)).cuda()  # B
        input = Variable(torch.LongTensor(input)).transpose(0,1).cuda()
        out = self.evaluate(self.encoder_source, self.decoder_target, input,
                                  input_length)
        noise = Variable(torch.LongTensor(noise)).cuda()
        noise_input = noise.float()+out.detach().float()
        loss_start1, adv_loss_start1 = self.train_step(encoder, decoder,noise_input.transpose(0,1),output_length,input,
                                                                                               input_length,
                                                                                               encoder_optimizer_source,
                                                                                               decoder_optimizer_source,
                                                                                               discriminator_optmizer,
                                                                                               criterion_adver,
                                                                                               target_adver)
        return (loss_start1, adv_loss_start1)






    def train_all(self,input_all):
        #TODO :  attention weights between the source and target decode are equal
        #Todo : sortir les loss et les backward a la fin
        # Initialize optimizers and criterion
        encoder_optimizer_source = self.encoder_optimizer_source
        decoder_optimizer_source = self.decoder_optimizer_source
        encoder_optimizer_target = self.encoder_optimizer_target
        decoder_optimizer_target = self.decoder_optimizer_target
        discriminator_optmizer=self.discriminator_optmizer
        all_batch=input_all
        criterion_adver = nn.BCELoss() #ou BCELoss ? et =criterion ?
        # Keep track of time elapsed and running averages
        input_length_debut = self.input_length_debut
        input_length_fin = self.input_length_fin
        #update_model
        # Get training data for this cycle
        self.encoder_source.train()
        self.decoder_source.train()
        self.encoder_target.train()
        self.encoder_target.train()
        all_histoire_debut_embedding = all_batch[0]
        all_histoire_fin_embedding = all_batch[1]
        all_histoire_debut_noise = all_batch[2]
        all_histoire_fin_noise = all_batch[3]
        all_debut_noise = all_batch[4]
        all_fin_noise = all_batch[5]
        #Start to start
        loss_auto_debut, adv_loss_auto_debut=self.train_auto_encoder(all_histoire_debut_embedding, all_histoire_debut_noise, input_length_debut, input_length_debut, True, encoder_optimizer_source,
                                                                                                                                                          decoder_optimizer_source, discriminator_optmizer, criterion_adver, self.encoder_source, self.decoder_source)
        # end to end
        loss_auto_fin, adv_loss_auto_fin = self.train_auto_encoder(all_histoire_fin_embedding, all_histoire_fin_noise, input_length_fin, input_length_fin, False, encoder_optimizer_target,
                                                                                                                                             decoder_optimizer_target, discriminator_optmizer, criterion_adver, self.encoder_target, self.decoder_target)
        #Start->End->Start
        loss_cross_debut, adv_cross_auto_debut=self.train_cross(all_histoire_debut_embedding, input_length_debut, input_length_fin, all_fin_noise, False,
                                                                                                   encoder_optimizer_source, decoder_optimizer_target,
                                                                                                   discriminator_optmizer, criterion_adver, self.encoder_source, self.decoder_target)
        # End->Start->End
        loss_cross_fin, adv_cross_auto_fin= self.train_cross(
            all_histoire_fin_embedding, input_length_fin, input_length_debut,
            all_debut_noise, True,
            encoder_optimizer_target, decoder_optimizer_source,
            discriminator_optmizer, criterion_adver, self.encoder_target, self.decoder_source)
        #compteurs
        main_loss_total = loss_auto_debut+loss_auto_fin+loss_cross_debut+loss_cross_fin
        discriminator_loss_total = adv_cross_auto_debut+adv_cross_auto_fin+adv_loss_auto_debut+adv_loss_auto_fin
        return (main_loss_total, loss_auto_debut, loss_auto_fin, loss_cross_debut, loss_cross_fin, discriminator_loss_total)

