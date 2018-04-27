from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import pickle



SOS_token = 0
EOS_token = 1
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s): 
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs():
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('corpus.txt', encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = []
    for l in lines:
        a, b0 = l.split('\t')
        b = normalizeString(b0)
        pairs.append((a, b))

    caption_list = Lang('caption_list')

    return caption_list, pairs

def prepareData():
    caption_list, pairs = readLangs()
    print("Read %s caption pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        caption_list.addSentence(pair[1])
    print("Counted words:")
    print(caption_list.name, caption_list.n_words)
    return caption_list, pairs

caption_list, pairs = prepareData()