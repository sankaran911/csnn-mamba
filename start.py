# -*- coding: utf-8 -*-
"""
@Author: Pangpd (https://github.com/pangpd/DS-pResNet-HSI)
@UsedBy: Katherine_Cao (https://github.com/Katherine-Cao/HSI_SNN)
"""

import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, accuracy_score

from utils import evaluate
import torch
import torch.nn.parallel

from utils.evaluate import AA_andEachClassAccuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):

    model.train()
    accs = np.ones((len(trainloader))) * -1000.0
    losses = np.ones((len(trainloader))) * -1000.0
    for batch_idx, (data,labels) in enumerate(trainloader):
        data = data

        data = np.transpose(data, (0, 2, 3, 1))
        hsi = data[..., 0:15]
        hsi = np.transpose(hsi, (0, 3, 1, 2))
        lidar = data[..., 15:]
        lidar = np.transpose(lidar, (0, 3, 1, 2))

        labels = labels
        if use_cuda:
            hsi = hsi.cuda()
            lidar = lidar.cuda()
            labels = labels.cuda()
    
        outputs = model(hsi,lidar)

        loss = criterion(outputs, labels)  # CrossEntropyloss

        losses[batch_idx] = loss.item()
        accs[batch_idx] = evaluate.accuracy(outputs.data, labels.data)[0].item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.average(losses), np.average(accs) 

    print("Train Loss =", np.average(losses))
    print("Train Accuracy =", np.average(accs))

def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    print(matrix)
    # OA, AA_mean, Kappa, AA = cal_results(matrix[1:,1:])
    # print("haha")
    # OA, AA_mean, Kappa, AA = cal_results(matrix[:,1:12])
    OA, AA_mean, Kappa, AA = cal_results(matrix[1:, 1:])

    return OA, AA_mean, Kappa, AA

def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        
        row_sum = np.sum(matrix[i, :])

        if row_sum == 0:
         AA[i] = 0
        else:
         AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA
def test(testloader, model, criterion, epoch, use_cuda):
    print("TEST FUNCTION ENTERED")
    print("TESTLOADER LEN =", len(testloader))

    model.eval()
    accs = np.ones((len(testloader))) * -1000.0
    losses = np.ones((len(testloader))) * -1000.0
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    with torch.no_grad():
        for batch_idx, (data,targets) in enumerate(testloader):
            print("TEST BATCH", batch_idx)
            data = data
            data = data.permute(0, 2, 3, 1)
            #print("Data shape:", data.shape)

            hsi = data[..., 0:15].permute(0, 3, 1, 2)
            lidar = data[..., 15:].permute(0, 3, 1, 2)
            targets = targets

            if use_cuda:
                hsi = hsi.cuda()
                lidar = lidar.cuda()
                targets = targets.cuda()

            outputs = model(hsi, lidar)

            #print("Batch =", batch_idx)
            #print(outputs.shape)

        
            losses[batch_idx] = criterion(outputs, targets).item()     # CrossEntropyLoss
            loss = criterion(outputs, targets)     # CrossEntropyLoss
            accs[batch_idx] = evaluate.accuracy(outputs.data, targets.data, topk=(1,))[0].item()
            
            #print("outputs shape =", outputs.shape)
            #print("targets shape =", targets.shape)


            prec1, t, p = accuracy(outputs, targets, topk=(1,))
           
            n = hsi.shape[0]
            objs.update(loss.data, n)
            top1.update(prec1[0].data, n)
            #print("Batch:", batch_idx)
            #print("t shape =", t.shape)
            #print("tar length =", len(tar))
            tar = np.append(tar, t.data.cpu().numpy())
            #print("After append tar length =", len(tar))
            pre = np.append(pre, p.data.cpu().numpy())
            #print("Final tar length =", len(tar))
            #print("Final pre length =", len(pre))

    return np.average(losses), np.average(accs),top1.avg,objs.avg,tar,pre

def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()
def predict(test_loader, model, use_cuda):
    #print("ENTERED PREDICT")

    model.eval()
    predicted = []

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            #print("PREDICT BATCH", batch_idx)

            data = data.permute(0,2,3,1)
            hsi = data[...,0:15].permute(0,3,1,2)
            lidar = data[...,15:].permute(0,3,1,2)

            if use_cuda:
                hsi = hsi.cuda()
                lidar = lidar.cuda()

            outputs = model(hsi, lidar)

            predicted.extend(outputs.cpu().numpy())

    #print("EXIT PREDICT")
    return np.array(predicted)

def adjust_learning_rate(optimizer, epoch, learn_rate):
    lr = learn_rate * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))  # 1-149:0.1，150-200:0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr