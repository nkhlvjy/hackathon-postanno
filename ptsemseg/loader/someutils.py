import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob

from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms

import re
if __name__ == '__main__' : 
    jsonpath = '/Users/aniketbhushan/Documents/Sandbox/image-seg/pytorch-semseg/data/res.json'
    with open(jsonpath) as small_file:
        res = json.load(small_file)
        labels = res['legend']
        labelToRGB = {}
        for key, val in labels.items() : 
            m = re.findall(r"rgb\((\d+), (\d+), (\d+)\)", val['rgb_color'])
            labelToRGB[key] = list(m[0])
            # print(key, list(m[0]))
        # print(labelToRGB)

    flatlistLabels = []
    flatListValues = []
    for key in labels.keys() :
        flatlistLabels.append(key)

    flatListValues = np.asarray(list(labelToRGB.values()))
    print(flatListValues.shape)
    print(flatlistLabels)

