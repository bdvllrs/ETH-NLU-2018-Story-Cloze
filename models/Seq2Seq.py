import torch
import torch.nn as nn
import torch.nn.functional as F
import math


USE_CUDA = torch.cuda.is_available()



class EncoderRNN(nn.Module):

    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super(EncoderRNN, self).__init__()
        """
        :param input_size
        :param embed_size
        :param hidden_size
        :param pretrained_weight
        :param n_layers
        :param dropout
        """
        # Define parameters
        self.input_size = input_size # V Taille Vocabulary (can be different, here not)
        self.hidden_size = hidden_size  # H
        self.embed_size = embed_size  # E
        self.n_layers = n_layers  # L (1 per default)
        self.dropout = dropout  # 0.5 per default
        # Define layers
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout,
                          bidirectional=True)  # Init (E,H,L, Bidirectionnel!)

    def forward(self, embedded, input_lengths, hidden=None):
        """
        :param input_seqs:
            Variable of shape (T,B), T is the number of words in the longuest sentence, B is Batchsize. Contening the indexing of the words reference to the voc
        :param input_lengths:
            list of integers (len=B) which reprensents the number of words in sequence for each batch. Normally Max(input_lengths)=T
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,H)
            last hidden stat of RNN(L*bidirectionnal,B,H)
        """
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded.float(),
                                                         input_lengths)  # cf doc pytorch : take embedding and input_length. Ready to go
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs

        return outputs, hidden  # (T,B,H),(L*bidirectionnal,B,H) | bidirectionnal=2 here


class Attn(nn.Module):
    def __init__(self, method, hidden_size, temporal=False):
        super(Attn, self).__init__()
        """
        :param method
        :param hidden_size
        :param temporal
        """
        # Define parameters
        self.method = method  # 2 methods cf publi
        self.hidden_size = hidden_size  # H
        self.temporal = temporal  # Temporal attention encoder
        self.softmax = nn.Softmax()
        # Define layers
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)  # Init(2*H,H)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, E_history=None):
        """
        :param hidden:
            (B,H)
        :param encoder_outputs:
            (T,B,H) can be also hidden decoder accumulation over time (t+1,B,H), depends on attention encoder or decoder
        :param E_history:
           Encoder history use only if intra temporal attention. Init with None, then (t,B,T)
        :returns:
            attn_energies which is alpha
        """
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(H, encoder_outputs)
        if self.temporal:
            if E_history is None:
                E_history = attn_energies.unsqueeze(0)
            else:
                E_history = torch.cat([E_history, attn_energies.unsqueeze(0)], 0)
                hist = E_history.view(-1, this_batch_size * max_len).t()
                attn_energies = self.softmax(hist)[:, -1].contiguous().view(this_batch_size, max_len)
            return F.softmax(attn_energies).unsqueeze(1), E_history
        else:
            # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
            return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):
        """
        :param hidden
        :param encoder_output
        """
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        elif self.method == 'concat':
            input = torch.cat([hidden, encoder_output], 2)
            energy = F.tanh(self.attn(input))  # [B*T*2H]->[B*T*H]
            energy = energy.transpose(2, 1)  # [B*H*T]
            v = self.v.repeat(encoder_output.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
            energy = torch.bmm(v, energy)  # [B*1*T]
            return energy.squeeze(1)  # [B*T]


class DecoderStep(nn.Module):
    def __init__(self, output_size, hidden_size, embed_size, n_layers, attention_bol=True, dropout_p=0.1):
        super(DecoderStep, self).__init__()
        """
        :param hidden_size
        :param embed_size
        :param output_size
        :param n_layers
        :param temporal
        :param de_att_bol
        :param point_bol
        :param attention_bol
        :param dropout_p
        """
        # Define parameters
        self.hidden_size = hidden_size  # H
        self.output_size = output_size  # V
        self.n_layers = n_layers  # L
        self.dropout_p = dropout_p  # 0.1 per default
        self.attention_bolean = attention_bol
        self.embed_size = embed_size

        # Define layers
        self.dropout = nn.Dropout(dropout_p)
        if self.attention_bolean:
            self.attn_encoder = Attn('concat', hidden_size)  # Init(methode score, H, bolean temporal), cf class
            self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers,
                              dropout=dropout_p)  # init(2*H+E,H,L)
            self.out = nn.Linear(hidden_size, embed_size)  # Wout(2H,V) case [1] and [2]
            self.out_proba = nn.Linear(hidden_size * 2, 1)
        else:
            self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout_p)  # init(E,H,L)
            self.out = nn.Linear(hidden_size, output_size)  # Wout(H,V) case [1] and [2]

    def forward(self, word_embedded, last_hidden, encoder_outputs):
        """
        :param word_input:
            tensor with SOS_Token length B
        :param last_hidden:
            Last hidden of the decoder, initialization with last hidden encoder (L,B,H)
        :param encoder_outputs:
            encoder output (T,B,H)
        :param E_hist:
            Encoder history use only if intra temporal attention. Init with None, then (1,B,T)
        :param t:
            number of time generate words [0 to max_length sequence]
        :param hd_history:
            hidden decoder accumulation over time (t+1,B,H)
        :param input_batches:
            input de l'encoder to be able to point to the entry (V,B)
        :returns:
            output decoder : max proba is the word to generate (B,V)
            hidden : will be last hidden next time
            alpha : to plot during the evaluation
            E_history : will be E_hist next time
            hd_history : will be hd_history next time
        """
        # Get the embedding of the current input word (last output word)
        #word_embedded = self.dropout(word_embedded)
        if self.attention_bolean:
            # Calculate attention weights -temporal or not- of encoder (alpha) and apply to encoder outputs (context_encoder)
            alpha = self.attn_encoder(last_hidden[-1], encoder_outputs)  # (B,1,T) alpha will be use later
            context_encoder = alpha.bmm(encoder_outputs.transpose(0, 1))  # (B,1,H)
            context_encoder = context_encoder.transpose(0, 1)  # (1,B,H) context with the encoder.
            context_encoder = context_encoder.squeeze(0)
            context_encoder = context_encoder.transpose(0,1)
            # attention on decoder and RNN input
            # Combine embedded input word and attended context, run through RNN
            rnn_input = torch.cat((word_embedded.float(), context_encoder), 0)
        else:
            rnn_input = word_embedded
        # RNN
        rnn_input=rnn_input.unsqueeze(0).transpose(1,2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output=self.out(output)
        if self.attention_bolean:
            context_encoder = context_encoder.squeeze(0)
            #output_concat = torch.cat((output, context_encoder), 1)
            output_concat=output
        else:
            alpha = None

        return output, output_concat, hidden, alpha



