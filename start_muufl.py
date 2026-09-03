# -*- coding: utf-8 -*-
"""
start_muufl.py
---------------
Train / test / predict helpers for the MUUFL Gulfport dataset
(official GatorSense scene-labeled benchmark).

Channel layout per sample (C, H, W):
    channels 0:HSI_CHANNELS                        -> HSI  (15 PCA components)
    channels HSI_CHANNELS:HSI_CHANNELS+LIDAR_CHANNELS -> LiDAR (1-2 elevation returns)

LIDAR_CHANNELS must match dataload_muufl.py's LIDAR_CHANNELS value (printed
at the end of that script's output) -- set it below accordingly.
"""

import numpy as np
from sklearn.metrics import confusion_matrix

from utils import evaluate
import torch
import torch.nn.parallel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HSI_CHANNELS = 15

# IMPORTANT: set this to match LIDAR_CHANNELS printed by dataload_muufl.py
# (the official file has 1 or 2 Lidar return structs -- check the printed
# "LiDAR raw shape" / "LIDAR_CHANNELS" line when you run dataload_muufl.py).
LIDAR_CHANNELS = 2


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def _split_hsi_lidar(data):
    """data: (B, C, H, W) -> hsi (B, HSI_CHANNELS, H, W), lidar (B, LIDAR_CHANNELS, H, W)"""
    data = np.transpose(data, (0, 2, 3, 1))           # (B, H, W, C)
    hsi = data[..., 0:HSI_CHANNELS]
    hsi = np.transpose(hsi, (0, 3, 1, 2))              # (B, HSI_CHANNELS, H, W)
    lidar = data[..., HSI_CHANNELS:HSI_CHANNELS + LIDAR_CHANNELS]
    lidar = np.transpose(lidar, (0, 3, 1, 2))          # (B, LIDAR_CHANNELS, H, W)
    return hsi, lidar


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    model.train()
    accs = np.ones((len(trainloader))) * -1000.0
    losses = np.ones((len(trainloader))) * -1000.0

    for batch_idx, (data, labels) in enumerate(trainloader):
        hsi, lidar = _split_hsi_lidar(data)

        #print("HSI:", hsi.shape)
        #print("LiDAR:", lidar.shape)

        if use_cuda:
            hsi = hsi.cuda()
            lidar = lidar.cuda()
            labels = labels.cuda()

       #print("Before model")
        outputs = model(hsi, lidar)
        #print("After model")
       
        #print("Unique labels:", torch.unique(labels))
        #print("Min label:", labels.min().item())
        #print("Max label:", labels.max().item())

        loss = criterion(outputs, labels)
        #print("Loss calculated")
        

        losses[batch_idx] = loss.item()
        accs[batch_idx] = evaluate.accuracy(outputs.data, labels.data)[0].item()

        optimizer.zero_grad()
        #print("Zero grad")

        loss.backward()
        #print("Backward done")

        optimizer.step()
        #print("Optimizer step done")

    return np.average(losses), np.average(accs)


def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    print(matrix)
    OA, AA_mean, Kappa, AA = cal_results(matrix[1:, 1:])
    return OA, AA_mean, Kappa, AA


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    total = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        row_sum = np.sum(matrix[i, :])
        AA[i] = 0 if row_sum == 0 else matrix[i, i] / row_sum
        total += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = total / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def test(testloader, model, criterion, epoch, use_cuda):
    model.eval()
    accs = np.ones((len(testloader))) * -1000.0
    losses = np.ones((len(testloader))) * -1000.0
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(testloader):
            print(f"TEST BATCH: {batch_idx}/{len(testloader)}")
            hsi, lidar = _split_hsi_lidar(data)

            if use_cuda:
                hsi = hsi.cuda()
                lidar = lidar.cuda()
                targets = targets.cuda()

            outputs = model(hsi, lidar)

            losses[batch_idx] = criterion(outputs, targets).item()
            loss = criterion(outputs, targets)
            accs[batch_idx] = evaluate.accuracy(outputs.data, targets.data, topk=(1,))[0].item()

            prec1, t, p = accuracy(outputs, targets, topk=(1,))
            n = hsi.shape[0]
            objs.update(loss.data, n)
            top1.update(prec1[0].data, n)
            tar = np.append(tar, t.data.cpu().numpy())
            pre = np.append(pre, p.data.cpu().numpy())

    return np.average(losses), np.average(accs), top1.avg, objs.avg, tar, pre


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()


def predict(test_loader, model, use_cuda):
    model.eval()
    predicted = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            hsi, lidar = _split_hsi_lidar(inputs)
            if use_cuda:
                hsi = hsi.cuda()
                lidar = lidar.cuda()
            [predicted.append(a) for a in model(hsi, lidar).data.cpu().numpy()]
    return np.array(predicted)


def adjust_learning_rate(optimizer, epoch, learn_rate):
    lr = learn_rate * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr