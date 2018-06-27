# encoding: utf-8

"""
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import h5py
import pdb
import IPython


class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None, target_transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)     
        label = torch.FloatTensor(label)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.image_names)

class ChestXrayDataSetHDF5(Dataset):
    def __init__(self, data_dir, set_idx, transform=None, target_transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        f = h5py.File(data_dir+'CXR8.h5','r')

        self.features = f['features']
        self.targets = f['targets']
        self.set_idx = set_idx
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        feature = self.features[self.set_idx[index]].reshape((224, 224, 1))
        feature = np.concatenate((feature, feature, feature), axis=2)

        target = self.targets[self.set_idx[index]]
        # IPython.embed()
        # pdb.set_trace()
        if self.transform is not None:
            feature = self.transform(feature)     
        target = torch.FloatTensor(target)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return feature, target

    def __len__(self):
        return len(self.set_idx)

class DDSMDataSetHDF5(Dataset):
    def __init__(self, data_dir, set_idx, transform=None, target_transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        f = h5py.File(data_dir+'DDSM_RGB.h5','r')

        self.features = f['features']
        self.targets = f['targets']
        self.set_idx = set_idx
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        feature0 = np.transpose(self.features[self.set_idx[index],0], (1,2,0))
        feature1 = np.transpose(self.features[self.set_idx[index],1], (1,2,0))
        feature2 = np.transpose(self.features[self.set_idx[index],2], (1,2,0))
        feature3 = np.transpose(self.features[self.set_idx[index],3], (1,2,0))

        target = self.targets[self.set_idx[index]]

        # IPython.embed()
        # pdb.set_trace()
        if self.transform is not None:
            feature0 = self.transform(feature0)     
            feature1 = self.transform(feature1)   
            feature2 = self.transform(feature2)   
            feature3 = self.transform(feature3)   
        # feature = torch.cat((feature0.reshape(1, 224, 224, 3),feature1.reshape(1, 224, 224, 3),feature2.reshape(1, 224, 224, 3),feature3).reshape(1, 224, 224, 3), 0)

        target = torch.FloatTensor(target)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # return feature0, np.argmax(target) #original
        return [feature0, feature1, feature2, feature3], np.argmax(target), self.set_idx[index] #modified by wawan fo parallel input
        # return target, self.set_idx[index]

    def __len__(self):
        return len(self.set_idx)

class DDSMDataSetHDF5_256(Dataset):
    def __init__(self, data_dir, set_idx, transform=None, target_transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        f = h5py.File(data_dir+'DDSM_RGB_256.h5','r')

        self.features = f['features']
        self.targets = f['targets']
        self.set_idx = set_idx
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        feature0 = np.transpose(self.features[self.set_idx[index],0], (1,2,0)).astype('uint8')
        feature1 = np.transpose(self.features[self.set_idx[index],1], (1,2,0)).astype('uint8')
        feature2 = np.transpose(self.features[self.set_idx[index],2], (1,2,0)).astype('uint8')
        feature3 = np.transpose(self.features[self.set_idx[index],3], (1,2,0)).astype('uint8')

        target = self.targets[self.set_idx[index]]

        # IPython.embed()
        # pdb.set_trace()
        if self.transform is not None:
            feature0 = self.transform(Image.fromarray(feature0))     
            feature1 = self.transform(Image.fromarray(feature1))    
            feature2 = self.transform(Image.fromarray(feature2))    
            feature3 = self.transform(Image.fromarray(feature3))    
        # feature = torch.cat((feature0.reshape(1, 224, 224, 3),feature1.reshape(1, 224, 224, 3),feature2.reshape(1, 224, 224, 3),feature3).reshape(1, 224, 224, 3), 0)

        target = torch.FloatTensor(target)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # return feature0, np.argmax(target) #original
        return [feature0, feature1, feature2, feature3], np.argmax(target), self.set_idx[index] #modified by wawan fo parallel input
        # return target, self.set_idx[index]

    def __len__(self):
        return len(self.set_idx)

class DDSMDataSetHDF5_256_FiveCrop(Dataset):
    def __init__(self, data_dir, set_idx, transform=None, target_transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        f = h5py.File(data_dir+'DDSM_RGB_256.h5','r')

        self.features = f['features']
        self.targets = f['targets']
        self.set_idx = set_idx
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        feature0 = np.transpose(self.features[self.set_idx[index%len(self.set_idx)],0], (1,2,0)).astype('uint8')
        feature1 = np.transpose(self.features[self.set_idx[index%len(self.set_idx)],1], (1,2,0)).astype('uint8')
        feature2 = np.transpose(self.features[self.set_idx[index%len(self.set_idx)],2], (1,2,0)).astype('uint8')
        feature3 = np.transpose(self.features[self.set_idx[index%len(self.set_idx)],3], (1,2,0)).astype('uint8')

        target = self.targets[self.set_idx[index%len(self.set_idx)]]

        # IPython.embed()
        # pdb.set_trace()
        if self.transform is not None:
            feature0 = self.transform(Image.fromarray(feature0))     
            feature1 = self.transform(Image.fromarray(feature1))    
            feature2 = self.transform(Image.fromarray(feature2))    
            feature3 = self.transform(Image.fromarray(feature3))    
        # feature = torch.cat((feature0.reshape(1, 224, 224, 3),feature1.reshape(1, 224, 224, 3),feature2.reshape(1, 224, 224, 3),feature3).reshape(1, 224, 224, 3), 0)

        target = torch.FloatTensor(target)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # return feature0, np.argmax(target) #original
        return [
            feature0[index//len(self.set_idx)], 
            feature1[index//len(self.set_idx)], 
            feature2[index//len(self.set_idx)], 
            feature3[index//len(self.set_idx)]
        ], np.argmax(target), [index%len(self.set_idx)] #modified by wawan fo parallel input
        # return target, self.set_idx[index]

    def __len__(self):
        return len(self.set_idx)*5

class DDSMDataSetHDF5_256_2Cls(Dataset):
    def __init__(self, data_dir, set_idx, transform=None, target_transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        f = h5py.File(data_dir+'DDSM_RGB_256.h5','r')

        self.features = f['features']
        self.targets = f['targets']
        self.set_idx = set_idx
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        feature0 = np.transpose(self.features[self.set_idx[index],0], (1,2,0)).astype('uint8')
        feature1 = np.transpose(self.features[self.set_idx[index],1], (1,2,0)).astype('uint8')
        feature2 = np.transpose(self.features[self.set_idx[index],2], (1,2,0)).astype('uint8')
        feature3 = np.transpose(self.features[self.set_idx[index],3], (1,2,0)).astype('uint8')

        target = np.concatenate(
            (np.sum(self.targets[self.set_idx[index%len(self.set_idx)]][0:3]).reshape((1,)),
                self.targets[self.set_idx[index%len(self.set_idx)]][3].reshape((1,))))

        # IPython.embed()
        # pdb.set_trace()
        if self.transform is not None:
            feature0 = self.transform(Image.fromarray(feature0))     
            feature1 = self.transform(Image.fromarray(feature1))    
            feature2 = self.transform(Image.fromarray(feature2))    
            feature3 = self.transform(Image.fromarray(feature3))    
        # feature = torch.cat((feature0.reshape(1, 224, 224, 3),feature1.reshape(1, 224, 224, 3),feature2.reshape(1, 224, 224, 3),feature3).reshape(1, 224, 224, 3), 0)

        target = torch.FloatTensor(target)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # return feature0, np.argmax(target) #original
        return [feature0, feature1, feature2, feature3], np.argmax(target), self.set_idx[index] #modified by wawan fo parallel input
        # return target, self.set_idx[index]

    def __len__(self):
        return len(self.set_idx)

class DDSMDataSetHDF5_256_FiveCrop_2Cls(Dataset):
    def __init__(self, data_dir, set_idx, transform=None, target_transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        f = h5py.File(data_dir+'DDSM_RGB_256.h5','r')

        self.features = f['features']
        self.targets = f['targets']
        self.set_idx = set_idx
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        feature0 = np.transpose(self.features[self.set_idx[index%len(self.set_idx)],0], (1,2,0)).astype('uint8')
        feature1 = np.transpose(self.features[self.set_idx[index%len(self.set_idx)],1], (1,2,0)).astype('uint8')
        feature2 = np.transpose(self.features[self.set_idx[index%len(self.set_idx)],2], (1,2,0)).astype('uint8')
        feature3 = np.transpose(self.features[self.set_idx[index%len(self.set_idx)],3], (1,2,0)).astype('uint8')

        target = np.concatenate(
            (np.sum(self.targets[self.set_idx[index%len(self.set_idx)]][0:3]).reshape((1,)),
                self.targets[self.set_idx[index%len(self.set_idx)]][3].reshape((1,))))

        # IPython.embed()
        # pdb.set_trace()
        if self.transform is not None:
            feature0 = self.transform(Image.fromarray(feature0))     
            feature1 = self.transform(Image.fromarray(feature1))    
            feature2 = self.transform(Image.fromarray(feature2))    
            feature3 = self.transform(Image.fromarray(feature3))    
        # feature = torch.cat((feature0.reshape(1, 224, 224, 3),feature1.reshape(1, 224, 224, 3),feature2.reshape(1, 224, 224, 3),feature3).reshape(1, 224, 224, 3), 0)

        target = torch.FloatTensor(target)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # return feature0, np.argmax(target) #original
        return [
            feature0[index//len(self.set_idx)], 
            feature1[index//len(self.set_idx)], 
            feature2[index//len(self.set_idx)], 
            feature3[index//len(self.set_idx)]
        ], np.argmax(target), [index%len(self.set_idx)] #modified by wawan fo parallel input
        # return target, self.set_idx[index]

    def __len__(self):
        return len(self.set_idx)*5