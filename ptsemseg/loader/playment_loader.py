from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import torch
import numpy as np
import re
import json

class PlaymentLoader (Dataset) :
    def __init__(self, root) : 

        self.root = root
        self.originalFolderPath  = os.path.join(self.root, "JPEGImages")
        self.annotatedFolderPath = os.path.join(self.root, "SegmentationClass")
        
        self.allImagesPath = []
        self.annotatedImagesPath = []
        self.n_classes = 0

        self.objectLabels = []
        self.labelToRGB = None
        self.pascalLabels = None

        for imageI in sorted(os.listdir(self.originalFolderPath)) :
            self.allImagesPath.append(os.path.join(self.originalFolderPath, imageI))

        for imageI in sorted(os.listdir(self.annotatedFolderPath)) :
            self.annotatedImagesPath.append(os.path.join(self.annotatedFolderPath, imageI))

        self.targetToClass = sorted(os.listdir(self.path)) # ['class0', 'class1']

        for targetNo, targetI in enumerate(self.targetToClass) :
            for imageI in sorted(os.listdir(os.path.join(self.path, targetI))) :
                self.allImagesPath.append(os.path.join(self.path, targetI, imageI))
                self.allTargets.append(targetNo)

        self.load_labels()

    
    def __len__(self) : 
        return len(self.allImagesPath)
    
    def __getitem__(self, index) :

        im  = Image.open(self.allImagesPath[index])
        lbl = Image.open(self.annotatedImagesPath[index])
        
        return im, lbl
    
    def load_labels(self) : 
        jsonpath = '/Users/aniketbhushan/Documents/Sandbox/image-seg/pytorch-semseg/data/res.json'
        with open(jsonpath) as file:
            res = json.load(file)
            labels = res['legend']
            labelToRGB = {}
            for key, val in labels.items() : 
                m = re.findall(r"rgb\((\d+), (\d+), (\d+)\)", val['rgb_color'])
                labelToRGB[key] = list(m[0])
        
            self.objectLabels = labels.keys()
            self.labelToRGB = labelToRGB
            self.pascalLabels = np.asarray(list(labelToRGB.values()))

            self.n_classes = len(self.objectLabels)


    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors

        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return self.pascalLabels
    

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image

        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb


if __name__ == '__main__' :
    dataset = PlaymentLoader('./data')
    customDatasetLoaderObject = DataLoader(
        dataset,
        batch_size = 4,
        num_workers = 8,
        shuffle = False
    )

    for num, (images, targets) in enumerate(customDatasetLoaderObject) :
        for imageI in images : 
            plt.imshow(imageI.numpy().transpose(1, 2, 0), cmap='gray')
            plt.show()
            print(images.shape, targets.shape)