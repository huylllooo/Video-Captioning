import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import random
import pickle

from data_loader import caption_list, pairs
from seq2seq import EncoderRNN, DecoderRNN
from prepare_train_data import variableFromId, variablesFromPair, SOS_token, EOS_token
from cnn_model import model

use_cuda = torch.cuda.is_available()

maxPool = nn.MaxPool1d(5, stride=2)

teacher_forcing_ratio = 0.5

MAX_LENGTH = 88

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = len(input_variable)
    target_length = target_variable.size()[0]

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # use_teacher_forcing = False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


loss_all = 0.0


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.005):
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    global loss_all
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # training_pairs = [variablesFromPair(random.choice(pairs))
    #                   for i in range(n_iters)]
    training_pairs = [variablesFromPair(pairs[i])
                      for i in range(n_iters,n_iters+5)]
    criterion = nn.NLLLoss()

    for iter in range(1, 5 + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        print(len(input_variable))
        target_variable = training_pair[1]

        if training_pair[2] == True:
            loss = 0.0
        else:
            loss = train(input_variable, target_variable, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (Batch_no: %d %d%%) %.4f' % (timeSince(start, iter / 5),
                                         n_iters/5, iter / 5 * 100, print_loss_avg))
            loss_all += print_loss_avg

            plot_loss_avg = plot_loss_total / plot_every
            plot_loss_total = 0



hidden_size = 2046
encoder1 = EncoderRNN(2046, hidden_size)
attn_decoder1 = DecoderRNN(hidden_size, caption_list.n_words)

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

encoder1.load_state_dict(torch.load('encoder.pt'))
attn_decoder1.load_state_dict(torch.load('decoder.pt'))

def evaluate(encoder, decoder, vid_ID,max_length=MAX_LENGTH):
    input_variable, empty = variableFromId(vid_ID)
    input_length = len(input_variable)
    encoder_hidden = encoder.initHidden()

    # encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    # encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        # encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(caption_list.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words
def evaluateRandomly(encoder, decoder):
    for i in range(850, 854):
    # for i in range(2000, 2050):
        pair = pairs[i]
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)


init_learning_rate = 0.01
decay_rate = 0.5
rate = init_learning_rate

loss_list = []
for n in range(6):
    loss_all = 0.0
    epoch_num = n+1
    for i in range(360):
        batch_no = i*5
        trainIters(encoder1, attn_decoder1, batch_no, print_every=5, learning_rate=0.005)
        if (i%30 == 0):
            print('recorded!')
            evaluateRandomly(encoder1, attn_decoder1)
            torch.save(encoder1.state_dict(), 'encoder.pt')
            torch.save(attn_decoder1.state_dict(), 'decoder.pt')

    torch.save(encoder1.state_dict(), 'encoder.pt')
    torch.save(attn_decoder1.state_dict(), 'decoder.pt')

    loss_epoch = loss_all / 360
    print('Avg. epoch loss: %f' % loss_epoch)
    loss_list.append(loss_epoch)
    print('Finished epoch %d' % epoch_num)
    print(loss_list)

print('Done')