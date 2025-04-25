# This file contains all the functions needed for model stipulation, model training and later model investigation.

# Possible extensions: Collect and plot gradients? Clip gradients?
#        Experiment with other modelling choices, optimizer choices.
#      Play around with scheduled sampling.
#      Add other forms of regularisation beyond dropout, perhaps weight constraints?
#


######################################

colab = False

seed = 42

PAD_token = 0
SOS_token = 1
EOS_token = 2

import sys
terminal_output = sys.stdout

######################################
#
# Modelling choices we make here:
#      optimizer choice, clip grad norm (not done), the rate of scheduled sampling and the decay schedule,
#      learning rate scheduling (not done).
#
######################################


import random
from statistics import stdev
from random import choice
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
plt.style.use('ggplot')
from collections import Counter
from sklearn.manifold import TSNE
import math
from matplotlib.ticker import FuncFormatter
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import datetime
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.distributions.categorical import Categorical

#rng = np.random.default_rng(seed)
random.seed(seed)
torch.manual_seed(seed)

#if colab:
#    terminal_output = sys.stdout
#else:
#     this outputs to terminal
#    terminal_output = open('/dev/stdout', 'w')

if colab:
    from google.colab import files

####################################################################################################
##
## Stipulating the model class. Here we also stipulate all the functions needed for beam search later.
##
####################################################################################################


class EncoderRNN(nn.Module):
    # This is a fairly standard RNN encoder. We use the packed_sequence functionality of PyTorch to save compute.
    # In calling self.gru, we do not provide the initial state, therefore it defaults to zeros. An alternative could be to
    #  institute the initial state as a trainable PyTorch parameter.
    # We divert from the Luong paper in two ways. First, we use GRUs, so as not to have the cell states to worry about.
    #   Second, we use a bidirectional encoder, setting the dimensionality to h_size/2. So, later, attention weights will be calculated
    #   wrt the bidirectional (concatenated) top-level GRU outputs. The decoder must be unidirectional. So what we'll do is
    #   initiate the decoder with dimensionality h_size. For this to work, we here rearrange h_out to concatenate both the forward and the
    #   backward final hidden states, to be able to feed into the decoder.
    #

    def __init__(self, vocab, h_size, n_layers, device, dropout=0):
        super(EncoderRNN, self).__init__()

        self.vocab = vocab
        self.h_size = h_size
        self.n_layers = n_layers
        self.device = device
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab, h_size)
        # the hidden size is halved:
        self.gru = nn.GRU(h_size, int(h_size / 2), n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths):
        embedded = self.embedding(input_seqs)
        packed = pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)
        # No initial hidden state to pass here.
        outputs, h = self.gru(packed, None)
        outputs, _ = pad_packed_sequence(outputs, padding_value=0)

        this_batch_size = input_seqs.size(1)
        # Rearrange h_out to concatenate both forward and backward final hidden state for each layer.
        h_out = torch.zeros(self.n_layers, this_batch_size, self.h_size, device=self.device)
        for i in range(self.n_layers):
            h_out[i] = torch.cat([h[i * 2], h[i * 2 + 1]], 1)

        return outputs, h_out


class Attn(nn.Module):
    #
    # This follows the definitions on page 3 of the Luong 2015 paper. Specifically, the "score" and the "alignment vector".
    # There is probably a way to use PyTorch broadcasting to make the two for-loops implementation in forward more elegant.
    # What we don't ensure, and what the paper stipulates, is that the alignment vector should equal the number of time steps
    # on the source side. Here, we make the alignment vector always equal to the longest (timewise) input sequence in the batch. And, because of how
    # softmax works, all its entries will be non-zero. It is left to the learning process to find out that attending to timesteps
    # corresponding to PAD in the input sequence (encoder hidden state = zero vector) is of no value.
    # mFF - we use the Linear layer without activation or bias, so it just does a matrix multiplication.
    #

    def __init__(self, method, hidden_size, device):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.device = device

        if self.method == 'general':
            self.mFF = nn.Linear(hidden_size, hidden_size, bias=False)

        elif self.method == 'concat':
            self.mFF = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            # initialize the below parameter vector using the same uniform distr. as for the weight matrix of self.mFF
            w = 1 / math.sqrt(float(hidden_size * 2))
            self.v = nn.Parameter(torch.rand(hidden_size) / (2 * w) - w)

    def score(self, dec_hidden, encoder_output):
        # Here both inputs are 1D arrays.
        # The score function always takes two vectors and returns a number.

        if self.method == 'dot':
            out = dec_hidden.dot(encoder_output)
            return out

        elif self.method == 'general':
            out = self.mFF(encoder_output)
            out = dec_hidden.dot(out)
            return out

        elif self.method == 'concat':
            out = self.mFF(torch.cat((dec_hidden, encoder_output)))
            out = self.v.dot(torch.tanh(out))
            return out

    def forward(self, dec_hidden, encoder_outputs):
        # vec x B     # time x B x vec (as usual)
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create the alignment vector
        # batch x max_encoder_timesteps
        alignment_vector = torch.zeros(this_batch_size, max_len, device=self.device)
        alignment_vector.to(self.device)

        # For each item-in-batch (of encoder outputs)
        for b in range(this_batch_size):
            # For each encoder timestep
            for t in range(max_len):
                alignment_vector[b, t] = self.score(dec_hidden[:, b], encoder_outputs[t, b])

        # So we return, in a batched form, the alignment vector, corresponding to (7), page 3 of the Luong paper.
        return F.softmax(alignment_vector, dim=1)


class LuongDecoder(nn.Module):
    #
    # This class runs one timestep of the decoder.
    # Again, here we have used GRU instead of LSTM to make the code cleaner.
    #

    def __init__(self, att_method, hidden_size, vocab_out, n_layers, device, dropout=0):
        super(LuongDecoder, self).__init__()

        self.att_method = att_method
        self.hidden_size = hidden_size
        self.vocab_out = vocab_out
        self.n_layers = n_layers
        self.device = device
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_out, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=False)
        self.joinerFF = nn.Linear(hidden_size * 2, hidden_size)
        self.projFF = nn.Linear(hidden_size, vocab_out)
        self.dropout_layer = nn.Dropout(dropout)
        self.alignment_vector = Attn(att_method, hidden_size, device)

    def forward(self, input_tokens, previous_hidden, encoder_outputs):
        # Here input_tokens is a batch of tokens of (dimensionality batch)

        embedded = self.embedding(input_tokens)
        embedded = embedded.unsqueeze(0)
        # this gives dimensionality (T=1, batch, vec)

        # Now the decoding RNN
        rnn_output, hidden = self.gru(embedded, previous_hidden)

        # use rnn_output in the attention calculation (the PyTorch implementation gives the topmost rnn_output in case n_layers > 1)
        #                                       1 x b x vec -> vec x b
        attn_weights = self.alignment_vector(rnn_output.squeeze(0).transpose(0, 1), encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)

        # b x 1 x time           # batch x time * vec
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.squeeze(1)
        # -> b x vec

        rnn_output = rnn_output.squeeze(0)  # -> b x vec

        # The RNN hidden state and the context vector now get concatenated, per equation (5)
        # and passed through an FC layer
        vecs = torch.cat((context, rnn_output), 1)
        # Could add dropout on this line as well? 
        vecs = torch.tanh(self.joinerFF(vecs))
        vecs = self.dropout_layer(vecs)

        # Just get the logits, no need to do the softmax.
        output = self.projFF(vecs)

        return output, hidden, attn_weights.squeeze(1)


def lossmaker1(x, y, model, device, epoch):
    # This is our loss function for a single batch.
    # This can be used both for training and for end-of-epoch evaluation
    # (done in other functions).

    loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

    #   we receive: (input_tokens, input_lengths), (output_tokens, output_lengths)
    input_tokens, input_lengths = x
    input_tokens = input_tokens.to(device)

    output_tokens, _ = y
    output_tokens = output_tokens.to(device)

    token_predictions, attentions = model(input_tokens, input_lengths, output_tokens, epoch)

    # prediction first, target second
    # L_out x B x vocab -> B x vocab x L_out
    L = loss(token_predictions.transpose(0, 1).transpose(1, 2), output_tokens.transpose(0, 1))

    return L

class Luong_full(nn.Module):
    #
    # This class runs through the encoder.
    # And runs through the decoder as well, one timestep at a time.
    #

    def __init__(self, vocab, h_size, n_layers, att_method, vocab_out, device, dropout, **kwargs):
        super(Luong_full, self).__init__()

        self.device = device
        self.vocab_out = vocab_out

        # Define layers
        self.encoder = EncoderRNN(vocab, h_size, n_layers, device, dropout)
        self.decoder = LuongDecoder(att_method, h_size, vocab_out, n_layers, device, dropout)

    def forward(self, input_tokens, input_lengths, output_tokens, epoch):
        # The output_tokens tensor will be used for calculating loss, so we need to know the time-length L of that tensor,
        # L is also needed for scheduled sampling

        L_out = output_tokens.size(0)
        b_size = output_tokens.size(1)
        L_input = input_tokens.size(0)

        # Create the first decoder input.
        decoder_input = torch.LongTensor([SOS_token] * b_size)
        decoder_input = decoder_input.to(self.device)

        # L_out x B x vocab
        all_decoder_outputs = torch.zeros(L_out, b_size, self.vocab_out, device=self.device)

        # L_out x batch x L_input
        all_attentions = torch.zeros(L_out, b_size, L_input, device=self.device)

        # the encoder step
        encoder_outputs, encoder_hidden = self.encoder(input_tokens, input_lengths)

        # initialize decoder_hidden
        decoder_hidden = encoder_hidden

        for t in range(L_out):
            # batch x vocab_out
            decoder_output, decoder_hidden, a_weights = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            all_decoder_outputs[t] = decoder_output
            all_attentions[t] = a_weights

            # always teacher-force:
            # teacher_forcing_param = 5

            # scheduled sampling:
            z_epoch = 7.0
            # when this goes below 0: never teacher-force
            teacher_forcing_param = (z_epoch - epoch) / z_epoch
            # now "flip a coin"
            # random.random() returns a random floating number between 0 and 1.
            if random.random() < teacher_forcing_param:
                # this is teacher forcing:
                decoder_input = output_tokens[t]
            else:
                # this is using model's own predictions
                # for each batch (row), sample according to the logits
                m = Categorical(logits=decoder_output)
                decoder_input = m.sample()

        # L_out x B x vocab & L_out, B, L_input
        return all_decoder_outputs, all_attentions

    def beam_decode(self, beam_width, input_tokens, max_dec_length):
        # Here we can only take a single input sequence, so input_lengths can be worked out.
        # We need to write this function here, because of the nontrivial ways encoder
        #    and decoder play out. We cannot just call model().
        # max_dec_length number of tokens we iterate for, upon decoding.
        #

        assert input_tokens.size(1) == 1, "More than one sequence input"

        input_tokens = input_tokens.to(self.device)
        input_lengths = [input_tokens.size(0)]

        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            # the encoder step
            encoder_outputs, encoder_hidden = self.encoder(input_tokens, input_lengths)

            # initialize decoder_hidden
            decoder_hidden = encoder_hidden

            # first decoder step:
            # 1 x vocab_out ; n_layers x 1 x h_size
            decoder_output, decoder_hidden, _ = self.decoder(torch.LongTensor([SOS_token]).to(self.device),
                                                             decoder_hidden, encoder_outputs)

            # now set up the beam of candidates
            log_probs, indices = F.log_softmax(decoder_output.squeeze(), dim=0).topk(beam_width)
            log_probs = log_probs.tolist()
            indices = indices.tolist()

            # We here institute the following nested list to represent the beam.
            #     The number of topmost elements in this list equals the number of items kept in the beam.

            #            [   [[token sequence],                   - list
            #                     [log_prob],               - float
            #                          hidden],             - torch tensor
            #                                 [], ...]

            # This format will change a bit, as we work out the possible continuations and append them (see below).

            beam = [[[indices[n]], [log_probs[n]], decoder_hidden] for n in range(beam_width)]

            for _ in range(max_dec_length - 1):

                # Run each beam element through the model to get beam_width number of possible continuations,
                # unless EOS has been reached.

                for k in range(beam_width):

                    #
                    sequence, log_prob, hidden = beam[k]

                    # we prepare the decoder input token
                    token = sequence[-1]

                    if token != EOS_token:
                        inp = torch.LongTensor([token]).to(self.device)

                        decoder_output, decoder_hidden, _ = self.decoder(inp,
                                                                         hidden, encoder_outputs)
                        log_probs, indices = F.log_softmax(decoder_output.squeeze(), dim=0).topk(beam_width)
                        log_probs = log_probs.tolist()
                        indices = indices.tolist()

                        # Now update the information stored in the k'th entry of the beam
                        #  to reflect the continuations we've just found.
                        sequence.append(indices)
                        beam[k] = [sequence, [log_prob[0] + d for d in log_probs], decoder_hidden]

                # print(beam)
                # print("*"*100)
                # reconfigure the beam to keep the best overall candidates
                beam = beam_reconfigure(beam)

        self.encoder.train()
        self.decoder.train()

        return beam

######################################
#
#  The below functions support the .beam_decode() method just above.
#
######################################


def beam_reconfigure(beam):
    # Takes the current beam with beam_width potential expansions appended (after calling the decoder)
    # and returns the beam_width entries with highest probability overall.
    # We should be able to accommodate seqs of varying length and the existence of the eos token.
    #

    # Build up the probabilities' list of lists with the following entries:
    # [candidate index, expansion index, log probability]

    beam_width = len(beam)
    probs = []
    for j in range(beam_width):
        _, log_prob, _ = beam[j]
        # Here, len(log_prob) = beam_width if the sequence got expanded.
        # But len(log_prob) = 1, if the sequence had reached EOS and there was no expansion. We accommodate both scenarios.
        for k in range(len(log_prob)):
            probs.append([j, k, log_prob[k]])

    # Now sort and pick only the beam_width candidates with the highest probability:
    probs = sorted(probs, key=lambda row: row[2], reverse=True)
    new_beam = []
    for l in range(beam_width):
        candidate_id, expansion_id, _ = probs[l]

        sequence, log_prob, hidden = beam[candidate_id]

        new_beam.append([clean_list(sequence, expansion_id), [log_prob[expansion_id]], hidden])

    return new_beam


def clean_list(l, index):
    # This is an auxiliary function that takes a sequence of tokens with possible continuations, e.g. [1, 2, 3, [4, 5, 6]]
    #  and returns an extended list with only the single continuation as indexed by the index variable.
    #

    last_element = l[-1]

    # First, do the case where there was no continuation, because EOS had been reached.
    if type(last_element) != list:
        assert index == 0
        return l

    last_element = last_element[index]

    out = l[:-1]
    out.append(last_element)

    return out




####################################################################################################
#
#  These functions implement the training loop in PyTorch. As well as plotting curves and saving models.
#
####################################################################################################

def scientific(x, pos):
    # x:  tick value - ie. what you currently see in yticks
    # pos: a position - ie. the index of the tick (from 0 to 9 in this example)
    return '%.1E' % x


def batcher(dataset, bsize):
    # go from a dataset object to batches without using a data loader
    indexes = range(0, len(dataset), bsize)
    for i in indexes:
        yield dataset[i:i+bsize]


def train_epoch(model, lossmaker, train_loader, optimizers, acc_steps, device, epoch):
    # runs through the training set once and trains the model using batches
    # collects and returns batch losses as a list

    model.train()
    i = 1
    loss_item = 0
    losses = []


    for x, y in train_loader:

        # reassignment ok, yes?
        # because we call .backward() and destroy computation graph?

        loss = lossmaker(x, y, model, device, epoch)/acc_steps

        loss.backward()
        loss_item = loss_item + loss.item()

        if i % acc_steps == 0:
            # clip gradients here
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
            # update model parameters
            [k.step() for k in optimizers]
            # erase gradients
            #[k.zero_grad() for k in optimizers]
            model.zero_grad(set_to_none=True)
            # save full_batch loss
            losses.append(loss_item)
            loss_item = 0

        i = i + 1

    return losses


def epochend_lcalc(model, lossmaker, loader, device, epoch):
    # this function returns the train or validation loss (e.g. at the end of an epoch, or for evaluation)
    # the loader given can be either train or val
    # here the model is in evaluation mode, so training loss could come out a bit shifted wrt batch evals (?)
    # you could code up e.g. accuracy here as well

    model.eval()

    with torch.no_grad():
        losses = []

        for x, y in loader:
            loss_item = lossmaker(x, y, model, device, epoch).item()
            losses.append(loss_item)

    # we average the batch losses, this requires loss additivity of the right kind
    return np.mean(losses)


def run_model(train_dataset, val_dataset, model_class, lossmaker, device,

              lrate, bsize, acc_steps, bsize_eval, epochs, patience = 1000, ratio = 1,

              # for saving BOTH the learning curve and the model
              save=False, path="", atlas = False, preloaded = False,

                # we pass these (= the rest) to the model class
                **kwargs

              ):
    # instantiate and run the model, return the final performance figure, plot the curves
    # this is meant to run well both on its own and within a loop for atlases or a loop for optimization
    # we leave explicit the parameters that will exist regardless of the model type
    # TODO: leave optimizer choice here?
    # LR scheduling?
    # you could return predictions as well!?

    global model_path, epoch

    # MAKE_MODEL_HERE
    if preloaded:
        m = model_class
    else:
        m = model_class(**kwargs, device = device).to(device)


    # truncate the dictionary:
    del kwargs['vocab']
    del kwargs['vocab_out']

    # t = "{:.0e}".format(lrate) + '_' + str(ratio) + "_"  + str(bsize * acc_steps) + '_' + str(kwargs).replace(" ", "")
    t = "{:.0e}".format(lrate) + '_' + str(ratio) + "_"  + str(bsize * acc_steps) + '_' + str(kwargs).replace(" ", "")
    modelpath = path + "/models"
    atlaspath = path + "/atlases"

    if colab:
        modelpath = '/content/gdrive/MyDrive/attention_seq2seq/' + modelpath
        atlaspath = '/content/gdrive/MyDrive/attention_seq2seq/' + atlaspath
        # print(modelpath)

    if save:
        #os.makedirs(path, exist_ok=True)
        os.makedirs(modelpath, exist_ok=True)
        os.makedirs(atlaspath, exist_ok=True)
        model_path = os.path.join(modelpath, t + ".pt")

        pic_path = os.path.join(path, t + ".png")
        if colab:
            pic_path = os.path.join('/content/gdrive/MyDrive/attention_seq2seq/' + path, t + ".png")
            #print(pic_path)
        if atlas:
            pic_path = os.path.join(atlaspath, t + ".png")


    start_time = datetime.datetime.now()


 #   train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=False)
 #   train_loader_epochend = DataLoader(train_dataset, batch_size=bsize_eval, shuffle=False)
 #   val_loader_epochend = DataLoader(val_dataset, batch_size=bsize_eval, shuffle=False)


    #optimizer = optim.SGD(m.parameters(), lr=lrate)
    #optimizers = [optim.Adam(m.parameters(), lr=lrate)]
    optimizers = [optim.Adam(m.encoder.parameters(), lr=lrate), optim.Adam(m.decoder.parameters(), lr=lrate)]
    optimizers = [optim.Adam(m.decoder.parameters(), lr=lrate), optim.Adam(m.decoder.parameters(), lr=lrate * ratio)]


    training_losses_full = []

    # these two are introduced for cleaner code: to be deleted later
    training_losses = [1000]
    val_losses = [1000]

    ref_val_loss = 1000
    trigger_times = 0
    patience = patience

    for epoch in range(epochs):

        print(datetime.datetime.now().strftime("%H:%M:%S") + "  Starting epoch " + str(epoch), file=terminal_output)

        # TRAINING
        training_losses_full += train_epoch(m, lossmaker, batcher(train_dataset, bsize), optimizers, acc_steps, device, epoch)

        print(datetime.datetime.now().strftime("%H:%M:%S") + "  Calculating figures", file=terminal_output)

        # END-OF-EPOCH loss figures

        train_loss = epochend_lcalc(m, lossmaker, batcher(train_dataset, bsize_eval), device, epoch)
        training_losses.append(train_loss)

        val_loss = epochend_lcalc(m, lossmaker, batcher(val_dataset, bsize_eval), device, epoch)

        # print(train_loss, val_loss)

        # FIRST: save model if current val_loss is the best so far
        if val_loss <= min(val_losses):
            if save:
               #2+3
                torch.save(m.state_dict(), model_path)

        # SECOND: do the early stopping thing
        if val_loss > ref_val_loss:
            trigger_times += 1
            if trigger_times == patience:
                val_losses.append(val_loss)

                print(datetime.datetime.now().strftime("%H:%M:%S") + "  Ending epoch " + str(epoch) +
                      "    train_loss: " + str(np.round(training_losses[-1], 2)) +
                      "   val_loss: " + str(np.round(val_losses[-1], 2)), file=terminal_output)
                print('Early stopping after completing epoch ' + str(epoch))
                break
            else:
                val_losses.append(val_loss)
        else:
            trigger_times = 0
            ref_val_loss = val_loss
            val_losses.append(val_loss)


            # THIRD: reduce learning rate
        #        if epoch > 10 and val_loss > previous val_loss -mindelta:
        #                lrate = lrate/2.0
        #                optimizers = [optim.Adam(m.encoder.parameters(), lr=lrate),
        #                              optim.Adam(m.decoder.parameters(), lr=lrate * ratio)]
        # print('reducing learning rate for next epoch')

        print(datetime.datetime.now().strftime("%H:%M:%S") + "  Ending epoch " + str(epoch) +
              "    train_loss: " + str(np.round(training_losses[-1], 2)) +
              "   val_loss: " + str(np.round(val_losses[-1], 2)), file=terminal_output)


    # here we could just save the last model, overriding early stopping
    # torch.save(m.state_dict(), os.path.join(modelpath, "L_"+kwargs['c'] + ".pt"))

    end_time = datetime.datetime.now()


    val_losses = val_losses[1:]
    training_losses = training_losses[1:]

    TLOSS = np.round(np.array(training_losses).min(), 4)
    VLOSS = np.round(np.array(val_losses).min(), 4)


    x = range(1, len(training_losses_full) + 1)

    # here we multiply an "epochs" array with the number of batch evaluations per epoch
    # dataset of 70 and bsize of 20 gives 4 batch evaluations per epoch
    # eps = np.arange(1, len(validation_losses) + 1) * math.ceil(len(train_dataset) / (bsize))
    # now modify for the gradient accumulation thing:
    # // integer division, / float division

    evals_per_epoch = math.ceil(len(train_dataset) / (bsize))//acc_steps
    #print("estimated")
    #print(evals_per_epoch)
    #print("actual")
    #print(len(training_losses_full)/len(val_losses))
    eps = np.arange(1, len(val_losses) + 1) * evals_per_epoch

    plt.style.use('default')
    rcParams['grid.linestyle'] = "dotted"
    rcParams['figure.figsize'] = 4, 4


    plt.plot(x, training_losses_full, 'r', label='tr_loss_full')  # , linewidth=0.8)
    plt.plot(eps, training_losses, 'o-', label='tr_loss', color="green", markersize=4)
    plt.plot(eps, val_losses, 'o-', label='val_loss', color="blue", markersize=4)
    plt.plot([], [], ' ', label="best tr_loss:\n" + str(TLOSS))
    plt.plot([], [], ' ', label="best val_loss:\n" + str(VLOSS))
    plt.plot([], [], ' ', label="duration:\n" + str(end_time - start_time)[0:4])

    plt.legend(fontsize=9)

    ###############################
    axes = plt.gca()
    # this is LR
    # axes.set_ylim([0, 0.5])

    axes.set_ylim([0, 3])
    # axes.set_ylim([0, 20])

    scientific_formatter = FuncFormatter(scientific)
    plt.gca().xaxis.set_major_formatter(scientific_formatter)
    plt.xticks(rotation=45)
    plt.grid(color='black')
    axes.set_axisbelow(False)


    plt.title(
        'lrate:' + "{:.0e}".format(lrate) + ', ratio:' + str(ratio) +  ', bsize:' + str(bsize*acc_steps) + ', \n'
        + 'n_epochs:' + str(epoch+1) + ', train:' + str(len(train_dataset)) + ', val:' + str(len(val_dataset)) + ', \n'
             + str(kwargs)[:40] + '\n' + str(kwargs)[40:],

        fontsize=9
    )

    plt.tight_layout()

    if save == True:
        plt.savefig(pic_path)
      #  if colab:
      #      p = os.path.normpath(pic_path)
      #      plt.savefig(p)
      #      files.download(p)

    print("", file=terminal_output)
    print("best training_loss = %s, best validation_loss = %s" % (TLOSS, VLOSS), file=terminal_output)
    print('Duration_: {}'.format(end_time - start_time), file=terminal_output)

    print("best training_loss = %s, best validation_loss = %s" % (TLOSS, VLOSS))
    print('Duration_: {}'.format(end_time - start_time))

    print(datetime.datetime.now().strftime("%H:%M:%S") + "  END RUN_MODEL CALL", file=terminal_output)
    print("", file=terminal_output)


    return (TLOSS, VLOSS)





