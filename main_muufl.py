# -*- coding: utf-8 -*-
"""
@CreatedDate:   2020/4/27 12:08
@Author: Pangpd(https://github.com/pangpd/DS-pResNet-HSI)
@UsedBy: lyh

main_muufl.py - uses the official GatorSense scene-labeled MUUFL benchmark
(muufl_gulfport_campus_1_hsi_220_label.mat), NOT the target-detection-only
MUUFL_TruthForSubImage.mat.

IMPORTANT: lidar_dim below must match LIDAR_CHANNELS reported by
dataload_muufl.py (1 or 2) -- and must also match LIDAR_CHANNELS set at
the top of start_muufl.py. Keep these three values in sync:
    dataload_muufl.py  -> printed LIDAR_CHANNELS
    start_muufl.py     -> LIDAR_CHANNELS constant
    main_muufl.py       -> lidar_dim argument to SNN(...)
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from dataload_muufl import (
    train_loader,
    test_loader,
    NUM_CLASSES,
    LIDAR_CHANNELS,
    class_dict,
    class_names_from_file,
    muufl_hsi,
    gt_MUUFL,test_coords
)
from utils.hyper_pytorch import *

import torch
import torch.nn.parallel
import warnings
warnings.filterwarnings('ignore')

from start_muufl import test, train, predict, output_metric
from complexNet_muufl import ComplexNet as SNN
from show_maps import show_pred,show_gt

np.set_printoptions(linewidth=400)
np.set_printoptions(threshold=sys.maxsize)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA AVAILABLE =", torch.cuda.is_available())
print("DEVICE =", device)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# -------------------------定义超参数--------------------------
data_path = os.path.join(os.getcwd(), 'data')  # 数据集路径

dataset = 'MUUFL'  # 数据集
seed = 1014
epochs = 1

learn_rate = 0.0085
momentum = 0.9
weight_decay = 0.0001
class_number = NUM_CLASSES   # pulled directly from the official label file (11)

iter = 1


def main():
    # ----------------------定义日志格式---------------------------
    time_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')
    log_path = os.path.join(os.getcwd(), "logs")  # logs目录
    log_dir = os.path.join(log_path, time_str)  # log组根目录

    torch.cuda.empty_cache()
    group_log_dir = os.path.join(log_dir, "Experiment_")  # logs组目录
    if not os.path.exists(group_log_dir):
        os.makedirs(group_log_dir)
    group_logger = None
    random_state = seed + iter
    print('-------------------------------------------Iter %s----------------------------------' % (iter + 1))
    start(group_log_dir, logger=group_logger)


def start(group_log_dir, logger):
    print('进入main.py 中的start方法！')
    use_cuda = True

    print(f"Building model with: input_dim=15, lidar_dim={LIDAR_CHANNELS}, num_cls={class_number}")

    # MUUFL: 15 PCA components for HSI, LIDAR_CHANNELS from official file, 11 classes
    print("LIDAR_CHANNELS =", LIDAR_CHANNELS)
    print("class_number =", class_number)

    model = SNN(25, input_dim=15, lidar_dim=LIDAR_CHANNELS, num_cls=class_number, fc_size=129600)

    if use_cuda:
        model = model.cuda()

    # 定义损失函数和优化器
    optimizer = torch.optim.SGD(model.parameters(), learn_rate, momentum=momentum,
                                 weight_decay=weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)
    criterion = torch.nn.CrossEntropyLoss()

    best_oa = -1
    best_aa = -1
    best_kappa = -1
    best_each_acc = -1
    best_acc = -1

    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

    # Class names pulled directly from the official label file (in class index order)
    class_names = [class_dict[i]["class_name"] for i in sorted(class_dict.keys())]

    train_start_time = time.time()
    for epoch in range(epochs):
        print("EPOCH =", epoch)

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        valid_loss, valid_acc, test_acc1, test_obj, tar_v, pre_v = test(
            test_loader, model, criterion, epoch, use_cuda)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)

        scheduler.step(valid_loss)

        OA_TE, AA_TE, Kappa_TE, CA_TE = output_metric(tar_v, pre_v)
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt

        cm = confusion_matrix(tar_v, pre_v)

        plt.figure(figsize=(8,8))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("confusion_matrix.png", dpi=300)
        plt.close()

        print("================================")
        print("Epoch %d | Train Loss: %.4f | Train Acc: %.2f%% | Test Loss: %.4f | Test Acc: %.2f%%" %
              (epoch, train_loss, train_acc, valid_loss, valid_acc))
        print("OA = %.2f%% | AA = %.2f%% | Kappa = %.4f" % (OA_TE * 100, AA_TE * 100, Kappa_TE))

        print("\nClassification Results Table")
        print("-" * 50)
        print("{:<5} {:<25} {:<10}".format("No.", "Class Name", "Accuracy"))
        for i, (cls, acc) in enumerate(zip(class_names, CA_TE * 100), start=1):
            print("{:<5} {:<25} {:.2f}%".format(i, cls, acc))
        print("-" * 50)
        print("================================")

        # save best model
        if valid_acc > best_acc:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': valid_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, group_log_dir + "/best_model_MUUFL.pth.tar")
            best_acc = valid_acc
            best_oa = OA_TE * 100
            best_aa = AA_TE * 100
            best_kappa = Kappa_TE
            best_each_acc = CA_TE * 100

    train_end_time = time.time()
    print("\nTraining complete. Total train time: %.2f s" % (train_end_time - train_start_time))
    print('BEST RESULTS -> OA: %.2f%% | AA: %.2f%% | Kappa: %.4f' % (best_oa, best_aa, best_kappa))
    for cls, acc in zip(class_names, best_each_acc):
        print(f"{cls}: {acc:.2f}%")

    if logger:
        logger.info('best_AA: %f, best_OA: %f, best_kappa: %f\n ' % (best_aa, best_oa, best_kappa))
        logger.info('best_CA: %s \n', best_each_acc)

    # ================= MAP GENERATION =================
    print("Generating Classification Map...")
    print("Generating Ground Truth Image...")
    show_gt(
        data=muufl_hsi,
        labels=gt_MUUFL,
        class_nums=11,
        save_path="ground_truth.png",
    )
    class_names=class_names_from_file

    print("Length =", len(class_names_from_file))

    
    pred = predict(test_loader, model, use_cuda)
    

    # Maximum probability (confidence) for each test sample
    confidence = np.max(pred, axis=1)

    # Create empty heat map
    heat_map = np.zeros(gt_MUUFL.shape, dtype=np.float32)

    # Fill confidence values at predicted pixel locations
    for (r, c), conf in zip(test_coords, confidence):
        heat_map[r, c] = conf

    # Plot heat map
    plt.figure(figsize=(8, 8))
    plt.imshow(heat_map, cmap='jet', vmin=0, vmax=1)
    plt.colorbar(label='Prediction Confidence')
    plt.title("QI-CSNN Confidence Heat Map (MUUFL)")
    plt.axis('off')

    plt.savefig(
        "results/MUUFL_Confidence_HeatMap.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
    print("Prediction shape =", pred.shape)

    pred_labels = np.argmax(pred, axis=1)


    full_pred = np.zeros(gt_MUUFL.shape, dtype=np.int64)

    for (r, c), label in zip(test_coords, pred_labels):
        full_pred[r, c] = label + 1

    print("Generating Prediction Map...")

    show_pred(
    full_pred,
    gt_MUUFL,
    NUM_CLASSES,
    "MUUFL",
    "results",
    True
)

    # ==================================================


def adjust_learning_rate(optimizer, epoch, learn_rate):
    lr = learn_rate * (0.1 ** (epoch // 50)) * (0.1 ** (epoch // 225))  # 每隔25个epoch更新学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()