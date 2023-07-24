# Copyright 2019 Karsten Roth and Biagio Brattoli
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


############################ LIBRARIES ######################################
import torch, os, numpy as np
import torch.nn as nn
import pretrainedmodels as ptm
import pretrainedmodels.utils as utils
import torchvision.models as models
import math
import torch
from SwinTransformer import SwinTransformer

"""============================================================="""
def initialize_weights(model):
    """
    Function to initialize network weights.
    NOTE: NOT USED IN MAIN SCRIPT.
    Args:
        model: PyTorch Network
    Returns:
        Nothing!
    """
    for idx,module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(0,0.01)
            module.bias.data.zero_()

"""=================================================================================================================================="""
### ATTRIBUTE CHANGE HELPER
def rename_attr(model, attr, name):
    """
    Rename attribute in a class. Simply helper function.
    Args:
        model:  General Class for which attributes should be renamed.
        attr:   str, Name of target attribute.
        name:   str, New attribute name.
    """
    setattr(model, name, getattr(model, attr))
    delattr(model, attr)


"""=================================================================================================================================="""
### NETWORK SELECTION FUNCTION
def networkselect(opt):
    """
    Selection function for available networks.
    Args:
        opt: argparse.Namespace, contains all training-specific training parameters.
    Returns:
        Network of choice
    """
    if opt.arch == 'resnet50':
        network =  ResNet50(opt)
    elif opt.arch == 'resnext50':
        network = ResNext50(opt)
    elif opt.arch == 'swinTransformer':
        network = SwinTransformer()
    elif opt.arch == 'convnext':
        network = ConvNeXt(opt)
    else:
        raise Exception('Network {} not available!'.format(opt.arch))

    if opt.init_pth:
        network.load_pth(opt.init_pth)
        print("Loaded: ", opt.init_pth)
        return network

    # initialize embedding layer
    if opt.embed_init == 'kaiming_normal':
        torch.nn.init.kaiming_normal_(network.model.last_linear.weight, mode='fan_in', nonlinearity='relu')
    elif opt.embed_init == 'kaiming_uniform':
        torch.nn.init.kaiming_uniform_(network.model.last_linear.weight, mode='fan_in', nonlinearity='relu')
    elif opt.embed_init == 'normal':
        network.model.last_linear.weight.data.normal_(0, 0.01)
        network.model.last_linear.bias.data.zero_()
    else:
        # do nothing, already intialized
        assert opt.embed_init == 'default'
    print(f"{opt.arch.upper()}: Embedding layer (last_linear) initialized with {opt.embed_init}")

    # finetune BatchNorm layers?
    if opt.ft_batchnorm:
        print(f"{opt.arch.upper()}: BatchNorm layers will be finetuned.")
    else:
        print(f"{opt.arch.upper()}: BatchNorm layers will NOT be finetuned.")
        for module in filter(lambda m: type(m) == nn.BatchNorm2d, network.modules()):
            module.eval()
            module.train = lambda _: None

    return network

class ResNet50(nn.Module):
    """
    Container for ResNet50 s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """
    def __init__(self, opt, list_style=False, no_norm=False):
        super(ResNet50, self).__init__()

        self.pars = opt

        if not opt.not_pretrained:
            print('ResNet50: Getting pretrained weights...')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
            print(self.model)
            print('ResNet50: Done.')
        else:
            print('ResNet50: Not utilizing pretrained weights!')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained=None)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)


    def forward(self, x, is_init_cluster_generation=False):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0),-1)

        mod_x = self.model.last_linear(x)
        #No Normalization is used if N-Pair Loss is the target criterion.
        return mod_x if self.pars.loss=='npair' else torch.nn.functional.normalize(mod_x, dim=-1)


    def to_optim(self, opt):
        if 'fc_lr_mul' in vars(opt).keys() and opt.fc_lr_mul!=0:
            return [{'params':self.model.conv1.parameters(),'lr':opt.lr,'weight_decay':opt.decay},
                    {'params':self.model.bn1.parameters(),'lr':opt.lr,'weight_decay':opt.decay},
                    {'params':self.model.layer1.parameters(),'lr':opt.lr,'weight_decay':opt.decay},
                    {'params':self.model.layer2.parameters(),'lr':opt.lr,'weight_decay':opt.decay},
                    {'params':self.model.layer3.parameters(),'lr':opt.lr,'weight_decay':opt.decay},
                    {'params':self.model.layer4.parameters(),'lr':opt.lr,'weight_decay':opt.decay},
                    {'params':self.model.last_linear.parameters(),'lr':opt.lr*opt.fc_lr_mul,'weight_decay':opt.decay}]
        else:
            return [{'params':self.parameters(),'lr':opt.lr,'weight_decay':opt.decay}]


"""============================================================="""

class ResNext50(nn.Module):
    """
    Container for ResNext50 s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """
    def __init__(self, opt, list_style=False, no_norm=False):
        super(ResNext50, self).__init__()

        self.pars = opt

        if not opt.not_pretrained:
            print('ResNext50: Getting pretrained weights...')
            self.model = models.resnext50_32x4d(pretrained=True)
            print('ResNext50: Done.')
        else:
            print('ResNext50: Not utilizing pretrained weights!')
            self.model = models.resnext50_32x4d(pretrained=None)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

        self.model.fc = torch.nn.Linear(self.model.fc.in_features, opt.embed_dim)


    def forward(self, x, is_init_cluster_generation=False):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0),-1)

        mod_x = self.model.fc(x)
        #No Normalization is used if N-Pair Loss is the target criterion.
        return mod_x if self.pars.loss=='npair' else torch.nn.functional.normalize(mod_x, dim=-1)


    def to_optim(self, opt):
        if 'fc_lr_mul' in vars(opt).keys() and opt.fc_lr_mul!=0:
            return [{'params':self.model.conv1.parameters(),'lr':opt.lr,'weight_decay':opt.decay},
                    {'params':self.model.bn1.parameters(),'lr':opt.lr,'weight_decay':opt.decay},
                    {'params':self.model.layer1.parameters(),'lr':opt.lr,'weight_decay':opt.decay},
                    {'params':self.model.layer2.parameters(),'lr':opt.lr,'weight_decay':opt.decay},
                    {'params':self.model.layer3.parameters(),'lr':opt.lr,'weight_decay':opt.decay},
                    {'params':self.model.layer4.parameters(),'lr':opt.lr,'weight_decay':opt.decay},
                    {'params':self.model.fc.parameters(),'lr':opt.lr*opt.fc_lr_mul,'weight_decay':opt.decay}]
        else:
            return [{'params':self.parameters(),'lr':opt.lr,'weight_decay':opt.decay}]

class ConvNeXt(nn.Module):
    """
    Container for ConvNeXt s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """
    def __init__(self, opt, list_style=False, no_norm=False):
        super(ConvNeXt, self).__init__()

        self.pars = opt

        if not opt.not_pretrained:
            print('ConvNeXt: Getting pretrained weights...')
            self.model = models.convnext_tiny(num_classes=1000, pretrained='imagenet')
            print('ConvNeXt: Done.')
        else:
            print('ConvNeXt: Not utilizing pretrained weights!')
            self.model = models.convnext_tiny(num_classes=1000, pretrained=None)

        print(self.model)
        self.model.classifier[2] = torch.nn.Linear(self.model.classifier[2].in_features, opt.embed_dim)

    def _forward_impl(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = self.model.classifier[0](x)
        x = self.model.classifier[1](x)
        x = self.model.classifier[2](x)
        return torch.nn.functional.normalize(x, dim=-1)

    def forward(self, x):
        return self._forward_impl(x)


    def to_optim(self, opt):
        if 'fc_lr_mul' in vars(opt).keys() and opt.fc_lr_mul!=0:
            return [{'params':self.model.features.parameters(),'lr':opt.lr,'weight_decay':opt.decay},
                    {'params':self.model.classifier.parameters(),'lr':opt.lr,'weight_decay':opt.decay},]
        else:
            return [{'params':self.parameters(),'lr':opt.lr,'weight_decay':opt.decay}]

