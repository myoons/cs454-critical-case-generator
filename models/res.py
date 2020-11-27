import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import torch.nn.functional as F

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)        
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:        
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:        
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine is not None:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)        

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)     
        
######################################################################   
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512): #512
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)] 
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu: 
            add_block += [nn.ReLU()]
        if dropout: 
            add_block += [nn.Dropout(p=0.5)] 
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x   

class Res18(nn.Module):
    def __init__(self):
        super(Res18, self).__init__()
        fea_dim = 5
        model_ft = models.resnet18(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.fc_embed = nn.Linear(512, fea_dim)
        self.fc_embed.apply(weights_init_classifier)
        self.classifier = ClassBlock(512, 5)
        self.classifier.apply(weights_init_classifier)
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        fea =  x.view(x.size(0), -1)
        embed_fea = self.fc_embed(fea)
        pred = self.classifier(fea)
        return pred 