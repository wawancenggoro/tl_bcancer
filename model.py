# encoding: utf-8

"""
The main CheXNet model implementation.
"""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from read_data import ChestXrayDataSet, ChestXrayDataSetHDF5
from sklearn.metrics import roc_auc_score
import pdb
import progressbar
import matplotlib.pyplot as plt 
import time
import IPython

CKPT_PATH = 'model.pth.tar'
N_CLASSES = 14
# HOME_DATA_DIR = './'
HOME_DATA_DIR = '/mnt/Storage/Projects/dataset/'
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = HOME_DATA_DIR + 'ChestX-ray14/images'
HDF5_DATA_DIR = HOME_DATA_DIR + 'ChestX-ray14/'
TRAIN_IMAGE_LIST = HOME_DATA_DIR + 'ChestX-ray14/labels/train_list.txt'
BATCH_SIZE = 16
NB_EPOCHS = 100

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def show_dataset():

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    trainDataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TRAIN_IMAGE_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(224),
                                        # normalize,
                                        # transforms.RandomHorizontalFlip
                                        transforms.TenCrop(224),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops])),
                                    ]))
    trainloader = DataLoader(dataset=trainDataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=8, pin_memory=True)

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        pdb.set_trace()

    # # Get a batch of training data
    # inputs, classes = next(iter(trainloader))
    # pdb.set_trace()

    # # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)

    # imshow(out, title=[class_names[x] for x in classes])

def inference():

    cudnn.benchmark = True

    # initialize and load the model
    model = DenseNet121(N_CLASSES).cuda()
    model = torch.nn.DataParallel(model).cuda()

    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH)
        # model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(checkpoint)
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    testDataset = ChestXrayDataSetHDF5(data_dir=HDF5_DATA_DIR,
                                    # image_list_file=TRAIN_IMAGE_LIST,
                                    transform=transforms.Compose([
                                        # transforms.Resize(224),
                                        # transforms.RandomHorizontalFlip(),
                                        RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize,
                                        # transforms.TenCrop(224),
                                        # transforms.Lambda
                                        # (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        # transforms.Lambda
                                        # (lambda crops: torch.stack([normalize(crop) for crop in crops])),
                                    ]),
                                    target_transform=None)
    test_loader = DataLoader(dataset=testDataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=8, pin_memory=True)

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    # switch to evaluate mode
    model.eval()

    for i, (inp, target) in enumerate(test_loader):
        target = target.cuda()
        gt = torch.cat((gt, target), 0)
        # bs, n_crops, c, h, w = inp.size()
        # input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
        input_var = torch.autograd.Variable(inp.cuda(), volatile=True)
        output = model(input_var)
        # output_mean = output.view(bs, n_crops, -1).mean(1)
        # pred = torch.cat((pred, output_mean.data), 0)
        pred = torch.cat((pred, output.data), 0)


    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))

def transformTargetTenCrop(label):
    return label.repeat(10,1)

def transformTargetNoTransform(label):
    return label.repeat(1,1)

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def RandomHorizontalFlip(object):
    if np.random.random() < 0.5:
        return flip_axis(object, 2)

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

def train():

    cudnn.benchmark = True

    # initialize and load the model
    model = DenseNet121(N_CLASSES).cuda()
    model = torch.nn.DataParallel(model).cuda()

    # if os.path.isfile(CKPT_PATH):
    #     print("=> loading checkpoint")
    #     checkpoint = torch.load(CKPT_PATH)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     print("=> loaded checkpoint")
    # else:
    #     print("=> no checkpoint found")

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    trainDataset = ChestXrayDataSetHDF5(data_dir=HDF5_DATA_DIR,
                                    # image_list_file=TRAIN_IMAGE_LIST,
                                    transform=transforms.Compose([
                                        # transforms.Resize(224),
                                        # transforms.RandomHorizontalFlip(),
                                        RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize,
                                        # transforms.TenCrop(224),
                                        # transforms.Lambda
                                        # (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        # transforms.Lambda
                                        # (lambda crops: torch.stack([normalize(crop) for crop in crops])),
                                    ]),
                                    target_transform=None)
    trainloader = DataLoader(dataset=trainDataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=8, pin_memory=True)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    for epoch in range(NB_EPOCHS):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0        
        # with progressbar.ProgressBar(max_value=2012) as bar:
        start_batch = time.time()
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            # bs, n_crops, c, h, w = inputs.size()
            # inputs = Variable(inputs.view(-1, c, h, w).cuda())
            # lb_bs, lb_n_crops, cls_size = labels.size()
            # labels = Variable(labels.view(-1, cls_size).cuda())

            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # pdb.set_trace()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # bar.update(i%100)

            # print statistics
            running_loss += loss.data[0]
            if i % 100 == 99:    # print every 2000 mini-batches
                end_batch = time.time()
                print('[%d, %5d] loss: %.3f run for %f s' %
                      (epoch + 1, i + 1, running_loss / 100, end_batch - start_batch))
                running_loss = 0.0
                start_batch = time.time()
        end = time.time()
        print('Time for epoch %d: %f s' % (epoch+1, end - start))
        torch.save(model.state_dict(), CKPT_PATH)

    print('Finished Training')

def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


if __name__ == '__main__':
    train()
    # inference()