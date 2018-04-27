import torchvision.models as models
import torch.nn as nn
import torch

#Load CNN Model (VGG-16)
model = models.vgg16(pretrained=True)
new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier = new_classifier

use_cuda = torch.cuda.is_available()
if use_cuda:
    model.cuda()
    print('CNN Model is using GPU')