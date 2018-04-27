import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import os
import io
from PIL import Image
from torchvision import transforms
import numpy as numpy
import cv2
import nltk

from cnn_model import model
from data_loader import caption_list, pairs

import glob
import re

use_cuda = torch.cuda.is_available()
maxPool = nn.MaxPool1d(5, stride=2)
# maxPool = nn.MaxPool1d(16, stride=16)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
    # transforms.Resize(256),
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

SOS_token = 0
EOS_token = 1

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

def imageListFromId(Id):
    image_list = []
    for filename in sorted(glob.glob('Frames/*.jpg'), key=numericalSort): #assuming gif
        if str(filename).startswith('Frames/' + Id):
            im=Image.open(filename)
            image_list.append(im)
    return image_list

def variableFromId(Id):
    image_list = imageListFromId(Id)
    if len(image_list) == 0:
        a = torch.autograd.Variable(torch.zeros(1, 1, 4096))
        b = maxPool(a)
        print(Id)
        if use_cuda:
            return b.cuda(), True
        else: 
            return b, True


    features_vectors = []
    n = 0
    count = 0
    for img in image_list:
        if n%5 == 0:
            img_tensor = preprocess(img)
            img_tensor.unsqueeze_(0)
            img_variable = Variable(img_tensor, requires_grad=False)
            if use_cuda:
                img_variable = img_variable.cuda(async=True)
            vector = model(img_variable)
            features_vectors.append(vector)
            count += 1
        n += 1
        if count > 4:
            break
    a = torch.stack(features_vectors)
    b = maxPool(a)
    # print(b.size())
    if use_cuda:
        return b.cuda(), False
    else: 
        return b, False


def variablesFromPair(pair):
    input_variable, isEmpty = variableFromId(pair[0])
    target_variable = variableFromSentence(caption_list, pair[1])
    return (input_variable, target_variable, isEmpty)