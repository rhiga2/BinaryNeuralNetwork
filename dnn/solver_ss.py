import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import numpy as np
import datasets.binary_data as binary_data
import datasets.stft as stft
import loss_and_metrics.bss_eval as bss_eval
import dnn.binary_layers as binary_layers
import visdom

class BinarySTFTSolver():
    def __init__(self, model, loss, optimizer=None, weighted=False,
        device=torch.device('cpu')):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.weighted = weighted
        self.loss = loss
        self.device = device
        self.stft = stft.STFT(nfft=1024, stride=256, win='hann').to(device)
        self.istft = stft.ISTFT(nfft=1024, stride=256, win='hann').to(device)
        self.bss_evaluate = bss_eval.BSSEvaluate(fs=8000).to(device)

    def _forward(self, binary_dl, raw_dl=None, clip_weights=False, train=False,
        get_baseline=False):
        running_loss = 0
        if raw_dl is not None:
            raw_dl = iter(raw_dl)
        for batch in binary_dl:
            if train:
                self.optimizer.zero_grad()
            bmag = batch['bmag'].to(self.device)
            ibm = batch['ibm'].to(self.device)
            bmag = bmag.to(self.device)
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
            cost.backward()
            if train:
                self.optimizer.step()
                if clip_weights:
                    self.model.clip_weights()
            bss_metrics = None
            if raw_dl is not None:
                raw_batch = next(raw_dl)
                bss_metrics = bss_eval.BSSMetricsList()
                mix = raw_batch['mix'].to(self.device)
                mix_mag, mix_phase = self.stft(mix)
                target = raw_batch['target']
                interference = raw_batch['interference']
                mask = binary_data.make_binary_mask(estimate)
                mask = mask.to(dtype=torch.float)
                mix_estimate = self.istft(mix_mag * mask, mix_phase)
                sources = torch.stack([target, interference], dim=1)
                sources = source.to(self.device)
                metrics = self.bss_evaluate(mix_estimate, sources)
                bss_metrics.extend(metrics)
        if train:
            self.optimizer.zero_grad()
        return running_loss / len(binary_dl.dataset), bss_metrics

    def train(self, dl, clip_weights=False):
        assert self.optimizer is not None
        self.model.train()
        return self._forward(dl, clip_weights=clip_weights)[0]

    def eval(self, dl, raw_dl):
        self.model.eval()
        return self._forward(dl, raw_dl=raw_dl)

    def get_baseline(self, dl, raw_dl):
        self.model.eval()
        return self._forward(dl, get_baseline=True)

class BinarySolver():
    def __init__(self, model, loss, optimizer=None, quantizer=None,
        classification=False, autoencode=False, device=torch.device('cpu')):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.quantizer = quantizer
        self.classification = classification
        self.autoencode = autoencode
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
        return running_loss / len(dl.dataset), bss_metrics

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
