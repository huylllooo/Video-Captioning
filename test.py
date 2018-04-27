from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

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
import nltk

use_cuda = torch.cuda.is_available()

maxPool = nn.MaxPool1d(5, stride=2)

MAX_LENGTH = 88

if use_cuda:
    model.cuda()
    print('CNN Model is using GPU')

def BLEU_score(gt_caption, sample_caption):
    """
    gt_caption: string, ground-truth caption
    sample_caption: string, your model's predicted caption
    Returns unigram BLEU score.
    """
    reference = []
    for caption in gt_caption:
        ref = [x for x in caption.split(' ') 
                     if ('<END>' not in x and '<START>' not in x and '<UNK>' not in x)]
        reference.append(ref)
    hypothesis = [x for x in sample_caption.split(' ') 
                  if ('<END>' not in x and '<START>' not in x and '<UNK>' not in x)]
    print(reference)
    BLEUscore = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0))
    b2 = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights=(0.5, 0.5, 0, 0))
    b3 = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights=(0.33, 0.33, 0.33, 0))
    b4 = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25))
    return BLEUscore, b2, b3, b4

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


######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#

def evaluateRandomly(encoder, decoder, n=3):
    total = 0.0
    for i in range(850, 871):
    # for i in range(2000, 2050):
        pair = pairs[i]
        print('>', pair[0])
        print('=', pair[1][0])
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        score,b2,b3,b4= BLEU_score(pair[1], output_sentence[:-6])
        print(score)
        print(b2)
        print(b3)
        print(b4)
        total += score
        print('')
    print('Avg. score is:')
    print(total/21)

hidden_size = 2046
encoder1 = EncoderRNN(2046, hidden_size)
attn_decoder1 = DecoderRNN(hidden_size, caption_list.n_words)

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

encoder1.load_state_dict(torch.load('encoder.pt'))
attn_decoder1.load_state_dict(torch.load('decoder.pt'))

######################################################################
#

evaluateRandomly(encoder1, attn_decoder1)

print('Done')