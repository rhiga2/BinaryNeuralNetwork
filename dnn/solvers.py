import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import numpy as np
import datasets.binary_data as binary_data
import datasets.stft as stft
import loss_and_metrics.bss_eval as bss_eval
import dnn.binary_layers as binary_layers
import dnn.bitwise_mlp as bitwise_mlp
import visdom

def get_dataloader_size(dl):
    if dl.sampler is not None:
        return len(dl.sampler)
    return len(dl.dataset)

class BinarySTFTSolver():
    def __init__(self, model, loss, optimizer=None, weighted=False,
        device=torch.device('cpu')):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.weighted = weighted
        self.loss = loss
        self.device = device
        self.bss_evaluate = bss_eval.BSSEvaluate(fs=8000).to(device)

    def _forward(self, binary_dl, raw_dl=None, clip_weights=False, train=False,
        get_baseline=False):
        running_loss = 0
        bss_metrics = bss_eval.BSSMetricsList()
        if raw_dl is not None:
            raw_dl = iter(raw_dl)
        for batch in binary_dl:
            if train:
                self.optimizer.zero_grad()
            bmag = batch['bmag'].to(torch.float32)
            ibm = batch['ibm'].to(torch.float32)
            bmag = bmag.to(self.device)
            ibm = ibm.to(self.device)
            bmag_size = bmag.size()
            bmag = 2*bmag - 1
            bmag = bitwise_mlp.flatten(bmag)
            estimate = ibm
            if not get_baseline:
                estimate = self.model(bmag)
                estimate = bitwise_mlp.unflatten(estimate, bmag_size[0],
                    bmag_size[2])
            # if self.weighted:
            #     cost = self.loss(estimate, ibm,
            #    weight=mix_mag/torch.std(mix_mag))
            # else:
            cost = self.loss(estimate, ibm)
            running_loss += cost.item() * bmag_size[0]
            if train:
                cost.backward()
                self.optimizer.step()
                if clip_weights:
                    self.model.clip_weights()

            if raw_dl is not None:
                raw_batch = next(raw_dl)
                mix = raw_batch['mix']
                mix_mag, mix_phase = stft(mix.numpy())
                target = raw_batch['target']
                interference = raw_batch['interference']
                mask = binary_data.make_binary_mask(estimate).detach()
                mask = mask.numpy().astype(dtype=np.float)
                mix_estimate = istft(mix_mag * mask, mix_phase)
                sources = torch.stack([target, interference], dim=1)
                sources = sources.to(self.device)
                mix_estimate = torch.FloatTensor(mix_estimate).to(self.device)
                metrics = self.bss_evaluate(mix_estimate, sources)
                bss_metrics.extend(metrics)
        if train:
            self.optimizer.zero_grad()
        dataset_size = get_dataloader_size(binary_dl)
        return running_loss / dataset_size, bss_metrics

    def train(self, dl, clip_weights=False):
        assert self.optimizer is not None
        self.model.train()
        return self._forward(dl, clip_weights=clip_weights, train=True)[0]

    def eval(self, dl, raw_dl):
        self.model.eval()
        return self._forward(dl, raw_dl=raw_dl)

    def get_baseline(self, dl, raw_dl):
        self.model.eval()
        return self._forward(dl, raw_dl=raw_dl, get_baseline=True)

class BinarySolver():
    def __init__(self, model, loss, optimizer=None, quantizer=None,
        classification=False, autoencode=False, device=torch.device('cpu')):
        self.model = model.to(device)
        self.loss = loss
        self.optimizer = optimizer
        self.quantizer = quantizer
        self.classification = classification
        self.autoencode = autoencode
        self.device = device
        self.bss_evaluate = bss_eval.BSSEvaluate(fs=8000).to(device)

    def _forward(self, dl, clip_weights=False, train=False,
        get_baseline=False):
        running_loss = 0
        bss_metrics = bss_eval.BSSMetricsList()
        for batch in dl:
            if train:
                self.optimizer.zero_grad()
            mix, target = batch['mixture'], batch['target']
            if self.autoencode:
                mix = target
            if self.quantizer:
                # mix = self.quantizer(mix).to(device=self.device, dtype=dtype)/255.
                target = self.quantizer(target).to(device=self.device, dtype=torch.long).view(-1)
            mix = mix.unsqueeze(1)
            mix = mix.to(self.device)
            target = target.to(self.device)
            estimate = target
            if not get_baseline:
                estimate = self.model(mix)

            if self.classification:
                estimate = estimate.permute(0, 2, 1).contiguous().view(-1, 256)
            else:
                estimate = estimate.squeeze(1)

            reconst_loss = self.loss(estimate, target)
            running_loss += reconst_loss.item() * mix.size(0)
            if train:
                reconst_loss.backward()
                self.optimizer.step()
                if clip_weights:
                    self.model.clip_weights()
            inter = batch['interference'].to(self.device)
            sources = torch.stack([target, inter], dim=1).to(self.device)
            metrics = self.bss_evaluate(estimate, sources)
            bss_metrics.extend(metrics)
        if train:
            self.optimizer.zero_grad()
        dataset_size = get_dataloader_size(dl)
        return running_loss / dataset_size, bss_metrics

    def train(self, dl, clip_weights=False):
        assert self.optimizer is not None
        self.model.train()
        return self._forward(dl, clip_weights=clip_weights, train=True)[0]

    def eval(self, dl):
        self.model.eval()
        return self._forward(dl)

    def get_baseline(self, dl):
        self.model.eval()
        return self._forward(dl, get_baseline=True)

class ImageRecognitionSolver():
    def __init__(self, model, loss, optimizer=None, flatten=False,
        device=torch.device('cpu')):
        self.model = model.to(device)
        self.loss = loss.to(device)
        self.flatten = flatten
        self.optimizer = optimizer
        self.device = device

    def _forward(self, dl, clip_weights=False, train=False):
        running_loss = 0
        running_accuracy = 0
        for batch_idx, (data, target) in enumerate(dl):
            if train:
                self.optimizer.zero_grad()
            if self.flatten:
                data = data.view(data.size(0), -1)
            data = data.to(self.device)
            target  = target.to(self.device)
            estimate = self.model(data)
            cost = self.loss(estimate, target)
            if train:
                cost.backward()
                self.optimizer.step()
                if clip_weights:
                    self.model.clip_weights()
            correct = torch.argmax(estimate, dim=1) == target
            running_accuracy += torch.sum(correct.float()).item()
            running_loss += cost.item() * data.size(0)
        if train:
            self.optimizer.zero_grad()
        dataset_size = get_dataloader_size(dl)
        return running_accuracy / dataset_size, running_loss / dataset_size

    def train(self, dl, clip_weights=False):
        assert self.optimizer is not None
        self.model.train()
        return self._forward(dl, clip_weights=clip_weights, train=True)

    def eval(self, dl):
        self.model.eval()
        return self._forward(dl, train=False)
