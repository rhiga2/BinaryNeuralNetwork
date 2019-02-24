import sys , os
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import glob

def get_data_from_directory(batchsize, dir, template='sample*.npz', return_dls=False):
    if dir[-1] != '/':
        dir += '/'
    trainset = DatasetFromDirectory(
        dir + 'train/', template = template
    )
    valset = DatasetFromDirectory(
        dir + 'val/', template = template
    )
    testset = DatasetFromDirectory(
        dir + 'test/', template = template
    )
    collate = lambda x: collate_and_trim(x, axis=0)
    if return_dls:
        train_dl = DataLoader(trainset, batch_size=batchsize, shuffle=True,
            collate_fn=collate)
        val_dl = DataLoader(valset, batch_size=batchsize, collate_fn=collate)
        test_dl = DataLoader(testset, batch_size=batchsize,
            collate_fn=collate)
        return train_dl, val_dl, test_dl
    return trainset, valset, testset

class DatasetFromDirectory():
    def __init__(self, data_dir, template='binary_data*.npz'):
        self.data_dir = data_dir
        if data_dir[-1] != '/':
            self.data_dir += '/'
        flist = glob.glob(self.data_dir + template)
        self.template = template.replace('*', '%d')
        self.length = len(flist)

    def __getitem__(self, i):
        fname = self.data_dir + (self.template%i)
        data = np.load(fname)
        return data

    def __len__(self):
        return self.length

def collate_and_trim(batch, axis=0, hop=1, dtype=torch.float):
    keys = list(batch[0].keys())
    outbatch = {key: [] for key in keys}
    min_length = min([sample[keys[0]].shape[axis] for sample in batch])
    for sample in batch:
        length = sample[keys[0]].shape[axis]
        start = (length - min_length) // 2
        for key in keys:
            indices = range(start, start+min_length)
            print(key, sample[key].take(indices=indices, axis=axis).shape)
            outbatch[key].append(sample[key].take(indices=indices, axis=axis))

    outbatch = {key: torch.as_tensor(np.stack(values, axis=0), dtype=dtype) for key, values in outbatch.items()}
    return outbatch
