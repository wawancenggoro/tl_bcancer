import argparse
import os
import shutil
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from densenet_4inputs import densenet121_4inputs

import numpy as np
import csv

from read_data import DDSMDataSetHDF5, DDSMDataSetHDF5_256, DDSMDataSetHDF5_256_FiveCrop
from read_data import DDSMDataSetHDF5_256_2Cls, DDSMDataSetHDF5_256_FiveCrop_2Cls
import IPython

CKPT_PATH = 'model.pth.tar'
# CKPT_BEST_PATH = 'model_best.pth.tar'
CKPT_BEST_PATH = 'checkpoint.pth.tar'
N_CLASSES = 4
N_CLASSES_CHEXNET = 14
HOME_DATA_DIR = './'
# HOME_DATA_DIR = '/mnt/Storage/DDSM/'
CLASS_NAMES = [ 'Benign_without_callbacks', 'Benigns', 'Cancers', 'Normal']
DATA_DIR = HOME_DATA_DIR
HDF5_DATA_DIR = HOME_DATA_DIR
TRAIN_INDEXES = HOME_DATA_DIR + 'DDSM_train_idx.csv'
VAL_INDEXES = HOME_DATA_DIR + 'DDSM_valid_idx.csv'
TEST_INDEXES = HOME_DATA_DIR + 'DDSM_test_idx.csv'
# TRAIN_INDEXES = HOME_DATA_DIR + 'DDSM_train_idx_small.csv'
# VAL_INDEXES = HOME_DATA_DIR + 'DDSM_train_idx_small.csv'
# TEST_INDEXES = HOME_DATA_DIR + 'DDSM_train_idx_small.csv'
BLOCK_CONFIG = (3, 12, 24, 16)

LAST_PARALLEL_BLOCKS = 1

TRAIN_PARALLEL_LAYER = False
TRAIN_PARALLEL_NORM_LAYER = True

TRANSFER_SINGLE_PATH_BLOCKS = True

TRANSFER_PARALLEL_NORM_LAYER = False
TRANSFER_SINGLE_NORM_LAYER = False

DROP_RATE = 0

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CheXNet Training')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

best_loss1 = float('inf')

class DenseNet121_4Inputs(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size, last_parallel_blocks, drop_rate):
        super(DenseNet121_4Inputs, self).__init__()
        self.densenet121 = densenet121_4inputs(
            last_parallel_blocks=last_parallel_blocks, 
            drop_rate=drop_rate)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x
        
class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = models.densenet121(pretrained=False)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x
        
def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

class RandomHorizontalFlip(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, feature):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if np.random.random() < 0.5:
            return flip_axis(feature, 2)
        return feature


def main():
    global args, best_loss1
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    model = DenseNet121_4Inputs(N_CLASSES, 
        last_parallel_blocks=LAST_PARALLEL_BLOCKS, drop_rate=DROP_RATE).cuda()
    model_source = DenseNet121(N_CLASSES_CHEXNET).cuda()
    
    model = torch.nn.DataParallel(model).cuda()
    model_source = torch.nn.DataParallel(model_source).cuda()
    
    # import IPython
    # IPython.embed()

    print("=> loading checkpoint")
    checkpoint = torch.load(CKPT_PATH)
    model_source.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint")    

    # import IPython
    # IPython.embed()

    ############################################################################################################################################################
    # Transfer Initial Convolution
    ############################################################################################################################################################    
    print("=> transferring and freezing initial convolution")

    model.module.densenet121.features0.conv0.weight.data.copy_(model_source.module.densenet121.features.conv0.weight.data)
    model.module.densenet121.features1.conv0.weight.data.copy_(model_source.module.densenet121.features.conv0.weight.data)
    model.module.densenet121.features2.conv0.weight.data.copy_(model_source.module.densenet121.features.conv0.weight.data)
    model.module.densenet121.features3.conv0.weight.data.copy_(model_source.module.densenet121.features.conv0.weight.data)

    model.module.densenet121.features0.conv0.weight.requires_grad = TRAIN_PARALLEL_LAYER
    model.module.densenet121.features1.conv0.weight.requires_grad = TRAIN_PARALLEL_LAYER
    model.module.densenet121.features2.conv0.weight.requires_grad = TRAIN_PARALLEL_LAYER
    model.module.densenet121.features3.conv0.weight.requires_grad = TRAIN_PARALLEL_LAYER

    if TRANSFER_PARALLEL_NORM_LAYER:
        model.module.densenet121.features0.norm0.weight.data.copy_(model_source.module.densenet121.features.norm0.weight.data)
        model.module.densenet121.features1.norm0.weight.data.copy_(model_source.module.densenet121.features.norm0.weight.data)
        model.module.densenet121.features2.norm0.weight.data.copy_(model_source.module.densenet121.features.norm0.weight.data)
        model.module.densenet121.features3.norm0.weight.data.copy_(model_source.module.densenet121.features.norm0.weight.data)

        model.module.densenet121.features0.norm0.bias.data.copy_(model_source.module.densenet121.features.norm0.bias.data)
        model.module.densenet121.features1.norm0.bias.data.copy_(model_source.module.densenet121.features.norm0.bias.data)
        model.module.densenet121.features2.norm0.bias.data.copy_(model_source.module.densenet121.features.norm0.bias.data)
        model.module.densenet121.features3.norm0.bias.data.copy_(model_source.module.densenet121.features.norm0.bias.data)

        model.module.densenet121.features0.norm0.running_mean.copy_(model_source.module.densenet121.features.norm0.running_mean)
        model.module.densenet121.features1.norm0.running_mean.copy_(model_source.module.densenet121.features.norm0.running_mean)
        model.module.densenet121.features2.norm0.running_mean.copy_(model_source.module.densenet121.features.norm0.running_mean)
        model.module.densenet121.features3.norm0.running_mean.copy_(model_source.module.densenet121.features.norm0.running_mean)

        model.module.densenet121.features0.norm0.running_var.copy_(model_source.module.densenet121.features.norm0.running_var)
        model.module.densenet121.features1.norm0.running_var.copy_(model_source.module.densenet121.features.norm0.running_var)
        model.module.densenet121.features2.norm0.running_var.copy_(model_source.module.densenet121.features.norm0.running_var)
        model.module.densenet121.features3.norm0.running_var.copy_(model_source.module.densenet121.features.norm0.running_var)

        model.module.densenet121.features0.norm0.weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
        model.module.densenet121.features1.norm0.weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
        model.module.densenet121.features2.norm0.weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
        model.module.densenet121.features3.norm0.weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER

        model.module.densenet121.features0.norm0.bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER
        model.module.densenet121.features1.norm0.bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER
        model.module.densenet121.features2.norm0.bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER
        model.module.densenet121.features3.norm0.bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER

    ############################################################################################################################################################
    # Transfer Dense Block 1
    ############################################################################################################################################################
    if LAST_PARALLEL_BLOCKS>=1:
        print("=> transferring and freezing dense block 1")

        for j in range(BLOCK_CONFIG[0]):
            if TRANSFER_PARALLEL_NORM_LAYER:
                for k in [0,3]:
                    model.module.densenet121.features0.denseblock1[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock1[j][k].weight.data)
                    model.module.densenet121.features1.denseblock1[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock1[j][k].weight.data)
                    model.module.densenet121.features2.denseblock1[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock1[j][k].weight.data)
                    model.module.densenet121.features3.denseblock1[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock1[j][k].weight.data)

                    model.module.densenet121.features0.denseblock1[j][k].bias.data.copy_(model_source.module.densenet121.features.denseblock1[j][k].bias.data)
                    model.module.densenet121.features1.denseblock1[j][k].bias.data.copy_(model_source.module.densenet121.features.denseblock1[j][k].bias.data)
                    model.module.densenet121.features2.denseblock1[j][k].bias.data.copy_(model_source.module.densenet121.features.denseblock1[j][k].bias.data)
                    model.module.densenet121.features3.denseblock1[j][k].bias.data.copy_(model_source.module.densenet121.features.denseblock1[j][k].bias.data)

                    model.module.densenet121.features0.denseblock1[j][k].running_mean.copy_(model_source.module.densenet121.features.denseblock1[j][k].running_mean)
                    model.module.densenet121.features1.denseblock1[j][k].running_mean.copy_(model_source.module.densenet121.features.denseblock1[j][k].running_mean)
                    model.module.densenet121.features2.denseblock1[j][k].running_mean.copy_(model_source.module.densenet121.features.denseblock1[j][k].running_mean)
                    model.module.densenet121.features3.denseblock1[j][k].running_mean.copy_(model_source.module.densenet121.features.denseblock1[j][k].running_mean)

                    model.module.densenet121.features0.denseblock1[j][k].running_var.copy_(model_source.module.densenet121.features.denseblock1[j][k].running_var)
                    model.module.densenet121.features1.denseblock1[j][k].running_var.copy_(model_source.module.densenet121.features.denseblock1[j][k].running_var)
                    model.module.densenet121.features2.denseblock1[j][k].running_var.copy_(model_source.module.densenet121.features.denseblock1[j][k].running_var)
                    model.module.densenet121.features3.denseblock1[j][k].running_var.copy_(model_source.module.densenet121.features.denseblock1[j][k].running_var)

                    model.module.densenet121.features0.denseblock1[j][k].weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
                    model.module.densenet121.features1.denseblock1[j][k].weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
                    model.module.densenet121.features2.denseblock1[j][k].weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
                    model.module.densenet121.features3.denseblock1[j][k].weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER

                    model.module.densenet121.features0.denseblock1[j][k].bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER
                    model.module.densenet121.features1.denseblock1[j][k].bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER
                    model.module.densenet121.features2.denseblock1[j][k].bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER
                    model.module.densenet121.features3.denseblock1[j][k].bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER

            for k in [2,5]:
                model.module.densenet121.features0.denseblock1[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock1[j][k].weight.data)
                model.module.densenet121.features1.denseblock1[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock1[j][k].weight.data)
                model.module.densenet121.features2.denseblock1[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock1[j][k].weight.data)
                model.module.densenet121.features3.denseblock1[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock1[j][k].weight.data)
                
                model.module.densenet121.features0.denseblock1[j][k].weight.requires_grad = TRAIN_PARALLEL_LAYER
                model.module.densenet121.features1.denseblock1[j][k].weight.requires_grad = TRAIN_PARALLEL_LAYER
                model.module.densenet121.features2.denseblock1[j][k].weight.requires_grad = TRAIN_PARALLEL_LAYER
                model.module.densenet121.features3.denseblock1[j][k].weight.requires_grad = TRAIN_PARALLEL_LAYER

        if LAST_PARALLEL_BLOCKS!=1:
            if TRANSFER_PARALLEL_NORM_LAYER:
                model.module.densenet121.features0.transition1.norm.weight.data.copy_(model_source.module.densenet121.features.transition1.norm.weight.data)
                model.module.densenet121.features1.transition1.norm.weight.data.copy_(model_source.module.densenet121.features.transition1.norm.weight.data)
                model.module.densenet121.features2.transition1.norm.weight.data.copy_(model_source.module.densenet121.features.transition1.norm.weight.data)
                model.module.densenet121.features3.transition1.norm.weight.data.copy_(model_source.module.densenet121.features.transition1.norm.weight.data)

                model.module.densenet121.features0.transition1.norm.bias.data.copy_(model_source.module.densenet121.features.transition1.norm.bias.data)
                model.module.densenet121.features1.transition1.norm.bias.data.copy_(model_source.module.densenet121.features.transition1.norm.bias.data)
                model.module.densenet121.features2.transition1.norm.bias.data.copy_(model_source.module.densenet121.features.transition1.norm.bias.data)
                model.module.densenet121.features3.transition1.norm.bias.data.copy_(model_source.module.densenet121.features.transition1.norm.bias.data)

                model.module.densenet121.features0.transition1.norm.running_mean.copy_(model_source.module.densenet121.features.transition1.norm.running_mean)
                model.module.densenet121.features1.transition1.norm.running_mean.copy_(model_source.module.densenet121.features.transition1.norm.running_mean)
                model.module.densenet121.features2.transition1.norm.running_mean.copy_(model_source.module.densenet121.features.transition1.norm.running_mean)
                model.module.densenet121.features3.transition1.norm.running_mean.copy_(model_source.module.densenet121.features.transition1.norm.running_mean)

                model.module.densenet121.features0.transition1.norm.running_var.copy_(model_source.module.densenet121.features.transition1.norm.running_var)
                model.module.densenet121.features1.transition1.norm.running_var.copy_(model_source.module.densenet121.features.transition1.norm.running_var)
                model.module.densenet121.features2.transition1.norm.running_var.copy_(model_source.module.densenet121.features.transition1.norm.running_var)
                model.module.densenet121.features3.transition1.norm.running_var.copy_(model_source.module.densenet121.features.transition1.norm.running_var)
            
                model.module.densenet121.features0.transition1.norm.weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
                model.module.densenet121.features1.transition1.norm.weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
                model.module.densenet121.features2.transition1.norm.weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
                model.module.densenet121.features3.transition1.norm.weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER

                model.module.densenet121.features0.transition1.norm.bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER
                model.module.densenet121.features1.transition1.norm.bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER
                model.module.densenet121.features2.transition1.norm.bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER
                model.module.densenet121.features3.transition1.norm.bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER

            # model.module.densenet121.features0.transition1.conv.weight.data.copy_(model_source.module.densenet121.features.transition1.conv.weight.data)
            # model.module.densenet121.features1.transition1.conv.weight.data.copy_(model_source.module.densenet121.features.transition1.conv.weight.data)
            # model.module.densenet121.features2.transition1.conv.weight.data.copy_(model_source.module.densenet121.features.transition1.conv.weight.data)
            # model.module.densenet121.features3.transition1.conv.weight.data.copy_(model_source.module.densenet121.features.transition1.conv.weight.data)

            # model.module.densenet121.features0.transition1.conv.weight.requires_grad = TRAIN_PARALLEL_LAYER
            # model.module.densenet121.features1.transition1.conv.weight.requires_grad = TRAIN_PARALLEL_LAYER
            # model.module.densenet121.features2.transition1.conv.weight.requires_grad = TRAIN_PARALLEL_LAYER
            # model.module.densenet121.features3.transition1.conv.weight.requires_grad = TRAIN_PARALLEL_LAYER

        model.module.densenet121.features0.denseblock1[2][2].weight.requires_grad = True    
        model.module.densenet121.features0.denseblock1[2][5].weight.requires_grad = True
        model.module.densenet121.features1.denseblock1[2][2].weight.requires_grad = True    
        model.module.densenet121.features1.denseblock1[2][5].weight.requires_grad = True
        model.module.densenet121.features2.denseblock1[2][2].weight.requires_grad = True    
        model.module.densenet121.features2.denseblock1[2][5].weight.requires_grad = True
        model.module.densenet121.features3.denseblock1[2][2].weight.requires_grad = True    
        model.module.densenet121.features3.denseblock1[2][5].weight.requires_grad = True
            
    elif TRANSFER_SINGLE_PATH_BLOCKS:
        print("=> transferring dense block 1")
        for j in range(BLOCK_CONFIG[0]):
            if TRANSFER_SINGLE_NORM_LAYER:
                for k in [0,3]:
                    model.module.densenet121.features.denseblock1[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock1[j][k].weight.data)
                    model.module.densenet121.features.denseblock1[j][k].bias.data.copy_(model_source.module.densenet121.features.denseblock1[j][k].bias.data)
                    model.module.densenet121.features.denseblock1[j][k].running_mean.copy_(model_source.module.densenet121.features.denseblock1[j][k].running_mean)
                    model.module.densenet121.features.denseblock1[j][k].running_var.copy_(model_source.module.densenet121.features.denseblock1[j][k].running_var)
            for k in [2,5]:
                model.module.densenet121.features.denseblock1[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock1[j][k].weight.data)
            #     model.module.densenet121.features.denseblock1[j][k].weight.requires_grad = False

            # model.module.densenet121.features.denseblock1[5][2].weight.requires_grad = True    
            # model.module.densenet121.features.denseblock1[5][5].weight.requires_grad = True

            # model.module.densenet121.features.denseblock1[4][2].weight.requires_grad = False    
            # model.module.densenet121.features.denseblock1[4][5].weight.requires_grad = False
            
        # if TRANSFER_SINGLE_NORM_LAYER:
        #     model.module.densenet121.features.transition1.norm.weight.data.copy_(model_source.module.densenet121.features.transition1.norm.weight.data)
        #     model.module.densenet121.features.transition1.norm.bias.data.copy_(model_source.module.densenet121.features.transition1.norm.bias.data)
        #     model.module.densenet121.features.transition1.norm.running_mean.copy_(model_source.module.densenet121.features.transition1.norm.running_mean)
        #     model.module.densenet121.features.transition1.norm.running_var.copy_(model_source.module.densenet121.features.transition1.norm.running_var)

        # model.module.densenet121.features.transition1.conv.weight.data.copy_(model_source.module.densenet121.features.transition1.conv.weight.data)

    # ############################################################################################################################################################
    # # Transfer Dense Block 2
    # ############################################################################################################################################################
    # if LAST_PARALLEL_BLOCKS>=2:
    #     print("=> transferring and freezing dense block 2")

    #     for j in range(BLOCK_CONFIG[1]):
    #         if TRANSFER_PARALLEL_NORM_LAYER:
    #             for k in [0,3]:
    #                 model.module.densenet121.features0.denseblock2[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock2[j][k].weight.data)
    #                 model.module.densenet121.features1.denseblock2[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock2[j][k].weight.data)
    #                 model.module.densenet121.features2.denseblock2[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock2[j][k].weight.data)
    #                 model.module.densenet121.features3.denseblock2[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock2[j][k].weight.data)

    #                 model.module.densenet121.features0.denseblock2[j][k].bias.data.copy_(model_source.module.densenet121.features.denseblock2[j][k].bias.data)
    #                 model.module.densenet121.features1.denseblock2[j][k].bias.data.copy_(model_source.module.densenet121.features.denseblock2[j][k].bias.data)
    #                 model.module.densenet121.features2.denseblock2[j][k].bias.data.copy_(model_source.module.densenet121.features.denseblock2[j][k].bias.data)
    #                 model.module.densenet121.features3.denseblock2[j][k].bias.data.copy_(model_source.module.densenet121.features.denseblock2[j][k].bias.data)

    #                 model.module.densenet121.features0.denseblock2[j][k].running_mean.copy_(model_source.module.densenet121.features.denseblock2[j][k].running_mean)
    #                 model.module.densenet121.features1.denseblock2[j][k].running_mean.copy_(model_source.module.densenet121.features.denseblock2[j][k].running_mean)
    #                 model.module.densenet121.features2.denseblock2[j][k].running_mean.copy_(model_source.module.densenet121.features.denseblock2[j][k].running_mean)
    #                 model.module.densenet121.features3.denseblock2[j][k].running_mean.copy_(model_source.module.densenet121.features.denseblock2[j][k].running_mean)

    #                 model.module.densenet121.features0.denseblock2[j][k].running_var.copy_(model_source.module.densenet121.features.denseblock2[j][k].running_var)
    #                 model.module.densenet121.features1.denseblock2[j][k].running_var.copy_(model_source.module.densenet121.features.denseblock2[j][k].running_var)
    #                 model.module.densenet121.features2.denseblock2[j][k].running_var.copy_(model_source.module.densenet121.features.denseblock2[j][k].running_var)
    #                 model.module.densenet121.features3.denseblock2[j][k].running_var.copy_(model_source.module.densenet121.features.denseblock2[j][k].running_var)

    #                 model.module.densenet121.features0.denseblock2[j][k].weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #                 model.module.densenet121.features1.denseblock2[j][k].weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #                 model.module.densenet121.features2.denseblock2[j][k].weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #                 model.module.densenet121.features3.denseblock2[j][k].weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER

    #         for k in [2,5]:
    #             model.module.densenet121.features0.denseblock2[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock2[j][k].weight.data)
    #             model.module.densenet121.features1.denseblock2[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock2[j][k].weight.data)
    #             model.module.densenet121.features2.denseblock2[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock2[j][k].weight.data)
    #             model.module.densenet121.features3.denseblock2[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock2[j][k].weight.data)

    #             model.module.densenet121.features0.denseblock2[j][k].weight.requires_grad = TRAIN_PARALLEL_LAYER
    #             model.module.densenet121.features1.denseblock2[j][k].weight.requires_grad = TRAIN_PARALLEL_LAYER
    #             model.module.densenet121.features2.denseblock2[j][k].weight.requires_grad = TRAIN_PARALLEL_LAYER
    #             model.module.densenet121.features3.denseblock2[j][k].weight.requires_grad = TRAIN_PARALLEL_LAYER

    #     if LAST_PARALLEL_BLOCKS!=2:
    #         if TRANSFER_PARALLEL_NORM_LAYER:
    #             model.module.densenet121.features0.transition2.norm.weight.data.copy_(model_source.module.densenet121.features.transition2.norm.weight.data)
    #             model.module.densenet121.features1.transition2.norm.weight.data.copy_(model_source.module.densenet121.features.transition2.norm.weight.data)
    #             model.module.densenet121.features2.transition2.norm.weight.data.copy_(model_source.module.densenet121.features.transition2.norm.weight.data)
    #             model.module.densenet121.features3.transition2.norm.weight.data.copy_(model_source.module.densenet121.features.transition2.norm.weight.data)

    #             model.module.densenet121.features0.transition2.norm.bias.data.copy_(model_source.module.densenet121.features.transition2.norm.bias.data)
    #             model.module.densenet121.features1.transition2.norm.bias.data.copy_(model_source.module.densenet121.features.transition2.norm.bias.data)
    #             model.module.densenet121.features2.transition2.norm.bias.data.copy_(model_source.module.densenet121.features.transition2.norm.bias.data)
    #             model.module.densenet121.features3.transition2.norm.bias.data.copy_(model_source.module.densenet121.features.transition2.norm.bias.data)

    #             model.module.densenet121.features0.transition2.norm.running_mean.copy_(model_source.module.densenet121.features.transition2.norm.running_mean)
    #             model.module.densenet121.features1.transition2.norm.running_mean.copy_(model_source.module.densenet121.features.transition2.norm.running_mean)
    #             model.module.densenet121.features2.transition2.norm.running_mean.copy_(model_source.module.densenet121.features.transition2.norm.running_mean)
    #             model.module.densenet121.features3.transition2.norm.running_mean.copy_(model_source.module.densenet121.features.transition2.norm.running_mean)

    #             model.module.densenet121.features0.transition2.norm.running_var.copy_(model_source.module.densenet121.features.transition2.norm.running_var)
    #             model.module.densenet121.features1.transition2.norm.running_var.copy_(model_source.module.densenet121.features.transition2.norm.running_var)
    #             model.module.densenet121.features2.transition2.norm.running_var.copy_(model_source.module.densenet121.features.transition2.norm.running_var)
    #             model.module.densenet121.features3.transition2.norm.running_var.copy_(model_source.module.densenet121.features.transition2.norm.running_var)

    #             model.module.densenet121.features0.transition2.norm.weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #             model.module.densenet121.features1.transition2.norm.weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #             model.module.densenet121.features2.transition2.norm.weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #             model.module.densenet121.features3.transition2.norm.weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER

    #             model.module.densenet121.features0.transition2.norm.bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #             model.module.densenet121.features1.transition2.norm.bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #             model.module.densenet121.features2.transition2.norm.bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #             model.module.densenet121.features3.transition2.norm.bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER

    #         model.module.densenet121.features0.transition2.conv.weight.data.copy_(model_source.module.densenet121.features.transition2.conv.weight.data)
    #         model.module.densenet121.features1.transition2.conv.weight.data.copy_(model_source.module.densenet121.features.transition2.conv.weight.data)
    #         model.module.densenet121.features2.transition2.conv.weight.data.copy_(model_source.module.densenet121.features.transition2.conv.weight.data)
    #         model.module.densenet121.features3.transition2.conv.weight.data.copy_(model_source.module.densenet121.features.transition2.conv.weight.data)

    #         model.module.densenet121.features0.transition2.conv.weight.requires_grad = TRAIN_PARALLEL_LAYER
    #         model.module.densenet121.features1.transition2.conv.weight.requires_grad = TRAIN_PARALLEL_LAYER
    #         model.module.densenet121.features2.transition2.conv.weight.requires_grad = TRAIN_PARALLEL_LAYER
    #         model.module.densenet121.features3.transition2.conv.weight.requires_grad = TRAIN_PARALLEL_LAYER

    # elif TRANSFER_SINGLE_PATH_BLOCKS:
    #     print("=> transferring dense block 2")
    #     for j in range(BLOCK_CONFIG[1]):
    #         if TRANSFER_SINGLE_NORM_LAYER:
    #             for k in [0,3]:
    #                 model.module.densenet121.features.denseblock2[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock2[j][k].weight.data)
    #                 model.module.densenet121.features.denseblock2[j][k].bias.data.copy_(model_source.module.densenet121.features.denseblock2[j][k].bias.data)
    #                 model.module.densenet121.features.denseblock2[j][k].running_mean.copy_(model_source.module.densenet121.features.denseblock2[j][k].running_mean)
    #                 model.module.densenet121.features.denseblock2[j][k].running_var.copy_(model_source.module.densenet121.features.denseblock2[j][k].running_var)
    #         for k in [2,5]:
    #             model.module.densenet121.features.denseblock2[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock2[j][k].weight.data)
    #         #     model.module.densenet121.features.denseblock2[j][k].weight.requires_grad = True

    #         # model.module.densenet121.features.denseblock2[23][2].weight.requires_grad = False    
    #         # model.module.densenet121.features.denseblock2[23][5].weight.requires_grad = False

    #     if TRANSFER_SINGLE_NORM_LAYER:
    #         model.module.densenet121.features.transition2.norm.weight.data.copy_(model_source.module.densenet121.features.transition2.norm.weight.data)
    #         model.module.densenet121.features.transition2.norm.bias.data.copy_(model_source.module.densenet121.features.transition2.norm.bias.data)
    #         model.module.densenet121.features.transition2.norm.running_mean.copy_(model_source.module.densenet121.features.transition2.norm.running_mean)
    #         model.module.densenet121.features.transition2.norm.running_var.copy_(model_source.module.densenet121.features.transition2.norm.running_var)

    #     model.module.densenet121.features.transition2.conv.weight.data.copy_(model_source.module.densenet121.features.transition2.conv.weight.data)

    # ############################################################################################################################################################
    # # Transfer Dense Block 3
    # ############################################################################################################################################################
    # if LAST_PARALLEL_BLOCKS>=3:
    #     print("=> transferring and freezing dense block 3")

    #     for j in range(BLOCK_CONFIG[2]):
    #         if TRANSFER_PARALLEL_NORM_LAYER:
    #             for k in [0,3]:
    #                 model.module.densenet121.features0.denseblock3[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock3[j][k].weight.data)
    #                 model.module.densenet121.features1.denseblock3[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock3[j][k].weight.data)
    #                 model.module.densenet121.features2.denseblock3[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock3[j][k].weight.data)
    #                 model.module.densenet121.features3.denseblock3[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock3[j][k].weight.data)

    #                 model.module.densenet121.features0.denseblock3[j][k].bias.data.copy_(model_source.module.densenet121.features.denseblock3[j][k].bias.data)
    #                 model.module.densenet121.features1.denseblock3[j][k].bias.data.copy_(model_source.module.densenet121.features.denseblock3[j][k].bias.data)
    #                 model.module.densenet121.features2.denseblock3[j][k].bias.data.copy_(model_source.module.densenet121.features.denseblock3[j][k].bias.data)
    #                 model.module.densenet121.features3.denseblock3[j][k].bias.data.copy_(model_source.module.densenet121.features.denseblock3[j][k].bias.data)

    #                 model.module.densenet121.features0.denseblock3[j][k].running_mean.copy_(model_source.module.densenet121.features.denseblock3[j][k].running_mean)
    #                 model.module.densenet121.features1.denseblock3[j][k].running_mean.copy_(model_source.module.densenet121.features.denseblock3[j][k].running_mean)
    #                 model.module.densenet121.features2.denseblock3[j][k].running_mean.copy_(model_source.module.densenet121.features.denseblock3[j][k].running_mean)
    #                 model.module.densenet121.features3.denseblock3[j][k].running_mean.copy_(model_source.module.densenet121.features.denseblock3[j][k].running_mean)

    #                 model.module.densenet121.features0.denseblock3[j][k].running_var.copy_(model_source.module.densenet121.features.denseblock3[j][k].running_var)
    #                 model.module.densenet121.features1.denseblock3[j][k].running_var.copy_(model_source.module.densenet121.features.denseblock3[j][k].running_var)
    #                 model.module.densenet121.features2.denseblock3[j][k].running_var.copy_(model_source.module.densenet121.features.denseblock3[j][k].running_var)
    #                 model.module.densenet121.features3.denseblock3[j][k].running_var.copy_(model_source.module.densenet121.features.denseblock3[j][k].running_var)

    #                 model.module.densenet121.features0.denseblock3[j][k].weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #                 model.module.densenet121.features1.denseblock3[j][k].weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #                 model.module.densenet121.features2.denseblock3[j][k].weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #                 model.module.densenet121.features3.denseblock3[j][k].weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER

    #                 model.module.densenet121.features0.denseblock3[j][k].bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #                 model.module.densenet121.features1.denseblock3[j][k].bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #                 model.module.densenet121.features2.denseblock3[j][k].bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #                 model.module.densenet121.features3.denseblock3[j][k].bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER

    #         for k in [2,5]:
    #             model.module.densenet121.features0.denseblock3[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock3[j][k].weight.data)
    #             model.module.densenet121.features1.denseblock3[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock3[j][k].weight.data)
    #             model.module.densenet121.features2.denseblock3[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock3[j][k].weight.data)
    #             model.module.densenet121.features3.denseblock3[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock3[j][k].weight.data)

    #             model.module.densenet121.features0.denseblock3[j][k].weight.requires_grad = TRAIN_PARALLEL_LAYER
    #             model.module.densenet121.features1.denseblock3[j][k].weight.requires_grad = TRAIN_PARALLEL_LAYER
    #             model.module.densenet121.features2.denseblock3[j][k].weight.requires_grad = TRAIN_PARALLEL_LAYER
    #             model.module.densenet121.features3.denseblock3[j][k].weight.requires_grad = TRAIN_PARALLEL_LAYER

    #     if LAST_PARALLEL_BLOCKS!=3:
    #         if TRANSFER_PARALLEL_NORM_LAYER:
    #             model.module.densenet121.features0.transition3.norm.weight.data.copy_(model_source.module.densenet121.features.transition3.norm.weight.data)
    #             model.module.densenet121.features1.transition3.norm.weight.data.copy_(model_source.module.densenet121.features.transition3.norm.weight.data)
    #             model.module.densenet121.features2.transition3.norm.weight.data.copy_(model_source.module.densenet121.features.transition3.norm.weight.data)
    #             model.module.densenet121.features3.transition3.norm.weight.data.copy_(model_source.module.densenet121.features.transition3.norm.weight.data)

    #             model.module.densenet121.features0.transition3.norm.bias.data.copy_(model_source.module.densenet121.features.transition3.norm.bias.data)
    #             model.module.densenet121.features1.transition3.norm.bias.data.copy_(model_source.module.densenet121.features.transition3.norm.bias.data)
    #             model.module.densenet121.features2.transition3.norm.bias.data.copy_(model_source.module.densenet121.features.transition3.norm.bias.data)
    #             model.module.densenet121.features3.transition3.norm.bias.data.copy_(model_source.module.densenet121.features.transition3.norm.bias.data)

    #             model.module.densenet121.features0.transition3.norm.running_mean.copy_(model_source.module.densenet121.features.transition3.norm.running_mean)
    #             model.module.densenet121.features1.transition3.norm.running_mean.copy_(model_source.module.densenet121.features.transition3.norm.running_mean)
    #             model.module.densenet121.features2.transition3.norm.running_mean.copy_(model_source.module.densenet121.features.transition3.norm.running_mean)
    #             model.module.densenet121.features3.transition3.norm.running_mean.copy_(model_source.module.densenet121.features.transition3.norm.running_mean)

    #             model.module.densenet121.features0.transition3.norm.running_var.copy_(model_source.module.densenet121.features.transition3.norm.running_var)
    #             model.module.densenet121.features1.transition3.norm.running_var.copy_(model_source.module.densenet121.features.transition3.norm.running_var)
    #             model.module.densenet121.features2.transition3.norm.running_var.copy_(model_source.module.densenet121.features.transition3.norm.running_var)
    #             model.module.densenet121.features3.transition3.norm.running_var.copy_(model_source.module.densenet121.features.transition3.norm.running_var)

    #             model.module.densenet121.features0.transition3.norm.weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #             model.module.densenet121.features1.transition3.norm.weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #             model.module.densenet121.features2.transition3.norm.weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #             model.module.densenet121.features3.transition3.norm.weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER

    #             model.module.densenet121.features0.transition3.norm.bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #             model.module.densenet121.features1.transition3.norm.bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #             model.module.densenet121.features2.transition3.norm.bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #             model.module.densenet121.features3.transition3.norm.bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER

    #         model.module.densenet121.features0.transition3.conv.weight.data.copy_(model_source.module.densenet121.features.transition3.conv.weight.data)
    #         model.module.densenet121.features1.transition3.conv.weight.data.copy_(model_source.module.densenet121.features.transition3.conv.weight.data)
    #         model.module.densenet121.features2.transition3.conv.weight.data.copy_(model_source.module.densenet121.features.transition3.conv.weight.data)
    #         model.module.densenet121.features3.transition3.conv.weight.data.copy_(model_source.module.densenet121.features.transition3.conv.weight.data)

    #         model.module.densenet121.features0.transition3.conv.weight.requires_grad = TRAIN_PARALLEL_LAYER
    #         model.module.densenet121.features1.transition3.conv.weight.requires_grad = TRAIN_PARALLEL_LAYER
    #         model.module.densenet121.features2.transition3.conv.weight.requires_grad = TRAIN_PARALLEL_LAYER
    #         model.module.densenet121.features3.transition3.conv.weight.requires_grad = TRAIN_PARALLEL_LAYER

    # elif TRANSFER_SINGLE_PATH_BLOCKS:
    #     print("=> transferring dense block 3")
    #     for j in range(BLOCK_CONFIG[2]):
    #         if TRANSFER_SINGLE_NORM_LAYER:
    #             for k in [0,3]:
    #                 model.module.densenet121.features.denseblock3[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock3[j][k].weight.data)
    #                 model.module.densenet121.features.denseblock3[j][k].bias.data.copy_(model_source.module.densenet121.features.denseblock3[j][k].bias.data)
    #                 model.module.densenet121.features.denseblock3[j][k].running_mean.copy_(model_source.module.densenet121.features.denseblock3[j][k].running_mean)
    #                 model.module.densenet121.features.denseblock3[j][k].running_var.copy_(model_source.module.densenet121.features.denseblock3[j][k].running_var)
    #         for k in [2,5]:
    #             model.module.densenet121.features.denseblock3[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock3[j][k].weight.data)

    #     if TRANSFER_SINGLE_NORM_LAYER:
    #         model.module.densenet121.features.transition3.norm.weight.data.copy_(model_source.module.densenet121.features.transition3.norm.weight.data)
    #         model.module.densenet121.features.transition3.norm.bias.data.copy_(model_source.module.densenet121.features.transition3.norm.bias.data)
    #         model.module.densenet121.features.transition3.norm.running_mean.copy_(model_source.module.densenet121.features.transition3.norm.running_mean)
    #         model.module.densenet121.features.transition3.norm.running_var.copy_(model_source.module.densenet121.features.transition3.norm.running_var)

    #     model.module.densenet121.features.transition3.conv.weight.data.copy_(model_source.module.densenet121.features.transition3.conv.weight.data)

    # ############################################################################################################################################################
    # # Transfer Dense Block 4
    # ############################################################################################################################################################
    # if LAST_PARALLEL_BLOCKS>=4:
    #     print("=> transferring and freezing dense block 4")

    #     for j in range(BLOCK_CONFIG[3]):
    #         if TRANSFER_PARALLEL_NORM_LAYER:
    #             for k in [0,3]:
    #                 model.module.densenet121.features0.denseblock4[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock4[j][k].weight.data)
    #                 model.module.densenet121.features1.denseblock4[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock4[j][k].weight.data)
    #                 model.module.densenet121.features2.denseblock4[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock4[j][k].weight.data)
    #                 model.module.densenet121.features3.denseblock4[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock4[j][k].weight.data)

    #                 model.module.densenet121.features0.denseblock4[j][k].bias.data.copy_(model_source.module.densenet121.features.denseblock4[j][k].bias.data)
    #                 model.module.densenet121.features1.denseblock4[j][k].bias.data.copy_(model_source.module.densenet121.features.denseblock4[j][k].bias.data)
    #                 model.module.densenet121.features2.denseblock4[j][k].bias.data.copy_(model_source.module.densenet121.features.denseblock4[j][k].bias.data)
    #                 model.module.densenet121.features3.denseblock4[j][k].bias.data.copy_(model_source.module.densenet121.features.denseblock4[j][k].bias.data)

    #                 model.module.densenet121.features0.denseblock4[j][k].running_mean.copy_(model_source.module.densenet121.features.denseblock4[j][k].running_mean)
    #                 model.module.densenet121.features1.denseblock4[j][k].running_mean.copy_(model_source.module.densenet121.features.denseblock4[j][k].running_mean)
    #                 model.module.densenet121.features2.denseblock4[j][k].running_mean.copy_(model_source.module.densenet121.features.denseblock4[j][k].running_mean)
    #                 model.module.densenet121.features3.denseblock4[j][k].running_mean.copy_(model_source.module.densenet121.features.denseblock4[j][k].running_mean)

    #                 model.module.densenet121.features0.denseblock4[j][k].running_var.copy_(model_source.module.densenet121.features.denseblock4[j][k].running_var)
    #                 model.module.densenet121.features1.denseblock4[j][k].running_var.copy_(model_source.module.densenet121.features.denseblock4[j][k].running_var)
    #                 model.module.densenet121.features2.denseblock4[j][k].running_var.copy_(model_source.module.densenet121.features.denseblock4[j][k].running_var)
    #                 model.module.densenet121.features3.denseblock4[j][k].running_var.copy_(model_source.module.densenet121.features.denseblock4[j][k].running_var)

    #                 model.module.densenet121.features0.denseblock4[j][k].weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #                 model.module.densenet121.features1.denseblock4[j][k].weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #                 model.module.densenet121.features2.denseblock4[j][k].weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #                 model.module.densenet121.features3.denseblock4[j][k].weight.requires_grad = TRAIN_PARALLEL_NORM_LAYER

    #                 model.module.densenet121.features0.denseblock4[j][k].bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #                 model.module.densenet121.features1.denseblock4[j][k].bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #                 model.module.densenet121.features2.denseblock4[j][k].bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER
    #                 model.module.densenet121.features3.denseblock4[j][k].bias.requires_grad = TRAIN_PARALLEL_NORM_LAYER

    #         for k in [2,5]:
    #             model.module.densenet121.features0.denseblock4[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock4[j][k].weight.data)
    #             model.module.densenet121.features1.denseblock4[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock4[j][k].weight.data)
    #             model.module.densenet121.features2.denseblock4[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock4[j][k].weight.data)
    #             model.module.densenet121.features3.denseblock4[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock4[j][k].weight.data)

    #             model.module.densenet121.features0.denseblock4[j][k].weight.requires_grad = TRAIN_PARALLEL_LAYER
    #             model.module.densenet121.features1.denseblock4[j][k].weight.requires_grad = TRAIN_PARALLEL_LAYER
    #             model.module.densenet121.features2.denseblock4[j][k].weight.requires_grad = TRAIN_PARALLEL_LAYER
    #             model.module.densenet121.features3.denseblock4[j][k].weight.requires_grad = TRAIN_PARALLEL_LAYER

    # elif TRANSFER_SINGLE_PATH_BLOCKS:
    #     print("=> transferring dense block 4")
    #     for j in range(BLOCK_CONFIG[3]):
    #         if TRANSFER_SINGLE_NORM_LAYER:
    #             for k in [0,3]:
    #                 model.module.densenet121.features.denseblock4[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock4[j][k].weight.data)
    #                 model.module.densenet121.features.denseblock4[j][k].bias.data.copy_(model_source.module.densenet121.features.denseblock4[j][k].bias.data)
    #                 model.module.densenet121.features.denseblock4[j][k].running_mean.copy_(model_source.module.densenet121.features.denseblock4[j][k].running_mean)
    #                 model.module.densenet121.features.denseblock4[j][k].running_var.copy_(model_source.module.densenet121.features.denseblock4[j][k].running_var)
    #         #         model.module.densenet121.features.denseblock4[j][k].weight.requires_grad = False

    #         # model.module.densenet121.features.denseblock4[15][2].weight.requires_grad = True    
    #         # model.module.densenet121.features.denseblock4[15][5].weight.requires_grad = True

    #         for k in [2,5]:
    #             model.module.densenet121.features.denseblock4[j][k].weight.data.copy_(model_source.module.densenet121.features.denseblock4[j][k].weight.data)
    
    ############################################################################################################################################################
    # Transfer Last Batch Normalization
    ############################################################################################################################################################
    # if TRANSFER_SINGLE_PATH_BLOCKS and TRANSFER_SINGLE_NORM_LAYER:
    #     print("=> transferring last batch norm")
    #     model.module.densenet121.features.norm5.weight.data.copy_(model_source.module.densenet121.features.norm5.weight.data)
    #     model.module.densenet121.features.norm5.bias.data.copy_(model_source.module.densenet121.features.norm5.bias.data)
    #     model.module.densenet121.features.norm5.running_mean.copy_(model_source.module.densenet121.features.norm5.running_mean)
    #     model.module.densenet121.features.norm5.running_var.copy_(model_source.module.densenet121.features.norm5.running_var)

    del(model_source)

    ############################################################################################################################################################
    # Train the Model
    ############################################################################################################################################################

    # define loss function (criterion) and optimizer
    # criterion = nn.BCELoss().cuda()
    
    import IPython
    IPython.embed()

    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), 
        lr=args.lr, betas=(0.9, 0.999), eps=1e-08, 
        weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # save_checkpoint({
    #     'epoch': 0,
    #     'arch': args.arch,
    #     'state_dict': model.state_dict(),
    #     'optimizer' : optimizer.state_dict(),
    # }, False)

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_idx = np.genfromtxt(TRAIN_INDEXES, delimiter=',').astype('int64')
    train_dataset = DDSMDataSetHDF5_256_FiveCrop_2Cls(data_dir=HDF5_DATA_DIR,
                                    set_idx = train_idx,
                                    transform=transforms.Compose([
                                        # transforms.CenterCrop(224),
                                        # transforms.ToTensor(),
                                        # normalize
                                        transforms.FiveCrop(224),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]),
                                    target_transform=None)

    # val_idx = np.genfromtxt(VAL_INDEXES, delimiter=',').astype('int64')
    # train_dataset = DDSMDataSetHDF5_256(data_dir=HDF5_DATA_DIR,
    #                                 set_idx = val_idx,
    #                                 transform=transforms.Compose([
    #                                     transforms.CenterCrop(224),
    #                                     transforms.ToTensor(),
    #                                     normalize
    #                                     # transforms.FiveCrop(224),
    #                                     # transforms.Lambda
    #                                     # (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    #                                     # transforms.Lambda
    #                                     # (lambda crops: torch.stack([normalize(crop) for crop in crops]))
    #                                 ]),
    #                                 target_transform=None)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, #(train_sampler is None),
        num_workers=0, pin_memory=False, sampler=None)

    
    val_idx = np.genfromtxt(VAL_INDEXES, delimiter=',').astype('int64')
    val_dataset = DDSMDataSetHDF5_256_2Cls(data_dir=HDF5_DATA_DIR,
                                    set_idx = val_idx,
                                    transform=transforms.Compose([
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize
                                        # transforms.FiveCrop(224),
                                        # transforms.Lambda
                                        # (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        # transforms.Lambda
                                        # (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]),
                                    target_transform=None)

    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # import IPython
    # IPython.embed()

    train_losses = []
    val_losses = []
    train_top1s = []
    val_top1s = []

    with open('performance.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(['epoch', 'train_loss', 'train_top1', 'val_loss', 'val_top1'])

    # import IPython
    # IPython.embed()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        (train_loss, train_top1)=train(train_loader, model, criterion, optimizer, epoch)
        train_losses.append(np.array(train_loss))
        train_top1s.append(np.array(train_top1))

        # evaluate on validation set
        (val_loss, val_top1) = validate(val_loader, model, criterion)
        val_losses.append(np.array(val_loss))
        val_top1s.append(np.array(val_top1))

        with open('performance.csv', 'a') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow([epoch, train_loss, train_top1, val_loss, val_top1])

        #loss
        plt.figure(1)
        plt.plot(train_losses, color='red')
        plt.plot(val_losses, color='green')
        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['learning', 'evaluation'], loc='upper right')
        # plt.show()
        plt.savefig('loss.png')

        #accuracy
        plt.figure(2)
        plt.plot(train_top1s, color='red')
        plt.plot(val_top1s, color='green')
        plt.title('Accuracy')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['learning', 'evaluation'], loc='lower right')
        # plt.show()
        plt.savefig('acc.png')        

        # import IPython
        # IPython.embed()

        # remember best prec@1 and save checkpoint
        is_best = val_loss < best_loss1
        best_loss1 = max(val_loss, best_loss1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, idx) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        # input_var = torch.autograd.Variable(input) #original
        input_var0 = torch.autograd.Variable(input[0]) #modified by wawan for parallel input
        input_var1 = torch.autograd.Variable(input[1]) #modified by wawan for parallel input
        input_var2 = torch.autograd.Variable(input[2]) #modified by wawan for parallel input
        input_var3 = torch.autograd.Variable(input[3]) #modified by wawan for parallel input
        target_var = torch.autograd.Variable(target)

        # bn_compare = nn.BatchNorm2d(112)
        # import IPython
        # IPython.embed()       
        # bn_compare.weight.data.copy_(model.module.densenet121.features0.norm0.weight.data)
        # bn_compare.running_mean.copy_(model.module.densenet121.features0.norm0.running_mean)
        # bn_compare.running_var.copy_(model.module.densenet121.features0.norm0.running_var)

        # bn_test = nn.BatchNorm2d(112)
        # bn_test.weight.data.copy_(model.module.densenet121.features0.norm0.weight.data)
        # bn_test.running_mean.copy_(model.module.densenet121.features0.norm0.running_mean)
        # bn_test.running_var.copy_(model.module.densenet121.features0.norm0.running_var)


        # compute output
        # output = model(input_var) #original
        # print(idx)
        # print(input_var0.max())
        output = model([input_var0, input_var1, input_var2, input_var3]) #modified by wawan for parallel input

        # import IPython
        # IPython.embed()

        loss = criterion(output, target_var)
        # print(output)
        # print('Train')


        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        top1.update(prec1[0].cpu()[0], input[0].size(0))
        losses.update(loss.data[0], input[0].size(0))

        # import IPython
        # IPython.embed()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #debug
        # print(input_var0.max())
        import math
        if math.isnan(model.module.densenet121.features0.norm0.running_mean[0]) or math.isnan(model.module.densenet121.features0.norm0.running_var[0]):
            print('NaN detected')
            import IPython
            IPython.embed()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))

            # import IPython
            # IPython.embed()
    return (losses.avg, top1.avg)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, idx) in enumerate(val_loader):
        target = target.cuda(async=True)
        # input_var = torch.autograd.Variable(input, volatile=True) #original
        input_var0 = torch.autograd.Variable(input[0], volatile=True) #modified by wawan for parallel input
        input_var1 = torch.autograd.Variable(input[1], volatile=True) #modified by wawan for parallel input
        input_var2 = torch.autograd.Variable(input[2], volatile=True) #modified by wawan for parallel input 
        input_var3 = torch.autograd.Variable(input[3], volatile=True) #modified by wawan for parallel input
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        # output = model(input_var) #original
        output = model([input_var0, input_var1, input_var2, input_var3]) #modified by wawan for parallel input
        loss = criterion(output, target_var)
        # print(output)
        # print('Valid')

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        top1.update(prec1[0].cpu()[0], input[0].size(0))
        losses.update(loss.data[0], input[0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

    return (losses.avg,top1.avg)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def inference():
    global args
    args = parser.parse_args()

    cudnn.benchmark = True

    # initialize and load the model
    model = DenseNet121_4Inputs(N_CLASSES, last_parallel_blocks=LAST_PARALLEL_BLOCKS, drop_rate=DROP_RATE).cuda()    
    model = torch.nn.DataParallel(model).cuda()

    if os.path.isfile(CKPT_BEST_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_BEST_PATH)
        model.load_state_dict(checkpoint['state_dict'])
        # model.load_state_dict(checkpoint)
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # idx = TRAIN_INDEXES
    idx = VAL_INDEXES
    # idx = TEST_INDEXES
    
    idx = np.genfromtxt(idx, delimiter=',').astype('int64')
    dataset = DDSMDataSetHDF5_256(data_dir=HDF5_DATA_DIR,
                                    set_idx = idx,
                                    transform=transforms.Compose([
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize
                                        # transforms.FiveCrop(224),
                                        # transforms.Lambda
                                        # (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        # transforms.Lambda
                                        # (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]),
                                    target_transform=None)

    loader = torch.utils.data.DataLoader(dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False)


    # import IPython
    # IPython.embed()

    # # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    # # switch to evaluate mode
    model.eval()

    for i, (inp, target, idxs) in enumerate(loader):
        target = target.cuda().float()
        gt = torch.cat((gt, target), 0) 

        input_var0 = torch.autograd.Variable(inp[0].cuda(), volatile=True)
        input_var1 = torch.autograd.Variable(inp[1].cuda(), volatile=True)
        input_var2 = torch.autograd.Variable(inp[2].cuda(), volatile=True)
        input_var3 = torch.autograd.Variable(inp[3].cuda(), volatile=True)

        output = model([input_var0, input_var1, input_var2, input_var3])

        pred = torch.cat((pred, output.data), 0)

    prec1 = accuracy(pred.long(), gt.long(), topk=(1,))
    print(prec1)

    np.savetxt('target.csv', gt.cpu().numpy(), delimiter=',') 
    np.savetxt('test.csv', pred.cpu().numpy(), delimiter=',') 

if __name__ == '__main__':
    main()
    # inference()
