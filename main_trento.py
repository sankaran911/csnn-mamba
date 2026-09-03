# -*- coding: utf-8 -*-
"""
@CreatedDate:   2020/4/27 12:08
@Author: Pangpd(https://github.com/pangpd/DS-pResNet-HSI)
@UsedBy: lyh

FIXES APPLIED (see comments marked with "# FIX:"):
  1. Added `import dataload_trento` so dataload_trento.gt_trento / .test_coords
     actually resolve (they were undefined before -> NameError).
  2. Cleaned up indentation so the classification table, prediction map, and
     new heat map all live inside the epoch loop consistently.
  3. Added a CONFIDENCE HEAT MAP (max softmax probability per pixel) in
     addition to the existing confusion matrix and discrete prediction map.
"""
import os
import sys
import time
import numpy as np

import dataload_trento  # FIX: needed for dataload_trento.gt_trento / test_coords
from dataload_trento import train_loader, test_loader
#from utils.auxiliary import save_acc_loss
#from utils.auxiliary import get_logger
from utils.hyper_pytorch import *
from datetime import datetime

import torch
import torch.nn.parallel
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
from start import test, train, predict, output_metric
from complexNet_trento import ComplexNet as SNN
# from models.complexNet_Trento_CSMS import ComplexNet as SNN
# from models.complexNet_Trento_2ccl import ComplexNet as SNN
# from models.complexNet_crossattention import ComplexNet as SNN
np.set_printoptions(linewidth=400)
np.set_printoptions(threshold=sys.maxsize)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA AVAILABLE =", torch.cuda.is_available())
print("DEVICE =", device)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# -------------------------定义超参数--------------------------
data_path = os.path.join(os.getcwd(), 'data')  # 数据集路径

dataset = 'PU'  # 数据集
seed = 1014
epochs = 1

learn_rate = 0.0085
# learn_rate = 0.00001
momentum = 0.9
weight_decay = 0.0001
# class_number = 22

iter = 1


def main():
    # ----------------------定义日志格式---------------------------
    time_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')
    log_path = os.path.join(os.getcwd(), "logs")  # logs目录
    log_dir = os.path.join(log_path, time_str)  # log组根目录

    oa_list = []
    aa_list = []
    kappa_list = []
    each_acc_list = []
    train_time_list = []
    test_time_list = []

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
    use_cuda = torch.cuda.is_available()

    # model = SNN(10, leak_mem=0.7, img_size=spatial_size, num_cls=class_number, input_dim=components) #SA最好精度在40个步长
    model = SNN(10, input_dim=15)  # SA最好精度在40个步长

    # print(model)
    if use_cuda:
        model = model.cuda()

    # 定义损失函数和优化器
    optimizer = torch.optim.SGD(model.parameters(), learn_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)
    criterion = torch.nn.CrossEntropyLoss()

    best_oa = -1
    best_aa = -1
    best_kappa = -1
    best_each_acc = -1
    best_acc = -1
    # 定义两个数组,记录训练损失和验证损失
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

    train_start_time = time.time()  # 返回当前的时间戳
    for epoch in range(epochs):
        print("EPOCH =", epoch)
        print("BEFORE TRAIN")

        print("CALLING TRAIN")
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        print("TRAIN RETURNED")
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_loss, valid_acc, test_acc1, test_obj, tar_v, pre_v = test(test_loader, model, criterion, epoch, use_cuda)
        print("AFTER TEST()")
        print("tar_v length =", len(tar_v))
        print("pre_v length =", len(pre_v))
        print("HELLO AFTER TEST")

        print("SKIPPED LOGGER")

        print("BEFORE OUTPUT_METRIC")
        OA_TE, AA_TE, Kappa_TE, CA_TE = output_metric(tar_v, pre_v)
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import numpy as np

        class_names = [
            "Apple Trees",
            "Buildings",
            "Ground",
            "Wood",
            "Vineyard",
            "Roads"
        ]

        # ================= Confusion Matrix =================
        cm = confusion_matrix(tar_v, pre_v)

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Trento Confusion Matrix")
        plt.colorbar()

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        # Write values inside each cell
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black"
                )

        plt.tight_layout()
        plt.savefig(os.path.join(group_log_dir, "Trento_Confusion_Matrix.png"), dpi=300)
        plt.close()
        print("AFTER OUTPUT_METRIC")

        print("================================")
        print("OA =", OA_TE * 100)
        print("AA =", AA_TE * 100)
        print("Kappa =", Kappa_TE)

        print("\nClassification Results Table")
        print("-" * 50)
        print("{:<5} {:<20} {:<10}".format("No.", "Class Name", "Accuracy"))

        for i, (cls, acc) in enumerate(zip(class_names, CA_TE * 100), start=1):
            print("{:<5} {:<20} {:.2f}%".format(i, cls, acc))

        print("-" * 50)
        print(f"OA = {OA_TE * 100:.2f}%")
        print(f"AA = {AA_TE * 100:.2f}%")
        print(f"Kappa = {Kappa_TE:.4f}")
        print("================================")

        # ================= Prediction =================
        pred = predict(test_loader, model, use_cuda)  # expected shape: (N, num_classes) logits or probs
        print("Prediction shape:", pred.shape)
        print("First prediction:", pred[0])

        # Convert raw scores to probabilities if they aren't already (softmax)
        pred_tensor = torch.from_numpy(pred) if isinstance(pred, np.ndarray) else pred
        pred_probs = F.softmax(pred_tensor, dim=1).cpu().numpy()

        pred_labels = np.argmax(pred_probs, axis=1) + 1          # +1 to reserve 0 for background
        confidence = np.max(pred_probs, axis=1)                  # max softmax prob per pixel
        print("Predicted labels shape:", pred_labels.shape)

        # ---------- Discrete class Prediction Map ----------
        prediction_map = np.zeros(dataload_trento.gt_trento.shape, dtype=np.uint8)
        for (r, c), label in zip(dataload_trento.test_coords, pred_labels):
            prediction_map[r, c] = label

        colors = np.array([
            [0, 0, 128],      # Background
            [255, 182, 193],  # Apple Trees
            [220, 20, 60],    # Buildings
            [255, 218, 185],  # Ground
            [139, 69, 19],    # Wood
            [255, 255, 0],    # Vineyard
            [173, 216, 230],  # Roads
            [0,255,0],         #class 7
        ], dtype=np.uint8)

        rgb = colors[prediction_map]
        print("Min label:", prediction_map.min())
        print("Max label:", prediction_map.max())
        print("Unique labels:", np.unique(prediction_map))

        plt.figure(figsize=(14, 4))
        plt.imshow(rgb)
        plt.title("Trento Prediction Map")
        plt.axis("off")
        plt.savefig(os.path.join(group_log_dir, "Trento_Prediction.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # ---------- Confidence Heat Map ----------
        # 0 for background/untested pixels, softmax confidence (0-1) elsewhere
        confidence_map = np.zeros(dataload_trento.gt_trento.shape, dtype=np.float32)
        for (r, c), conf in zip(dataload_trento.test_coords, confidence):
            confidence_map[r, c] = conf

        # mask background so it doesn't wash out the colormap
        masked_confidence = np.ma.masked_where(confidence_map == 0, confidence_map)

        plt.figure(figsize=(14, 4))
        cmap = plt.cm.jet
        cmap.set_bad(color="black")  # background stays black
        im = plt.imshow(masked_confidence, cmap=cmap, vmin=0, vmax=1)
        plt.title("Trento Prediction Confidence Heat Map")
        plt.axis("off")
        cbar = plt.colorbar(im, fraction=0.03, pad=0.02)
        cbar.set_label("Softmax Confidence")
        plt.savefig(os.path.join(group_log_dir, "Trento_Heatmap.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # ================= Save best model =================
        if valid_acc > best_acc:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': valid_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, group_log_dir + "/best_model.pth_Trento.tar")
            best_acc = valid_acc
            best_oa = OA_TE * 100
            best_aa = AA_TE * 100
            best_kappa = Kappa_TE
            best_each_acc = CA_TE * 100

    if logger:
        logger.info('best_AA: %f, best_OA: %f, best_kappa: %f\n ' % (best_aa, best_oa, best_kappa))
        logger.info('best_CA: %s \n', best_each_acc)

    # train_end_time = time.time()
    # checkpoint = torch.load(group_log_dir + "/best_model.pth_Trento.tar")
    # best_acc = checkpoint['best_acc']
    # start_epoch = checkpoint['epoch']
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    #
    # # 测试
    # test_start_time = time.time()
    # test_loss, test_acc, test_acc1, test_obj, tar_v, pre_v = test(test_loader, model, criterion, epoch, use_cuda)
    # OA_TE, AA_TE, Kappa_TE, CA_TE = output_metric(tar_v, pre_v)
    # print("OA: {:.2f} | AA: {:.2f} | Kappa: {:.4f}".format(OA_TE * 100, AA_TE * 100, Kappa_TE))
    # logger.info('AA: %f, OA: %f, kappa: %f\n '% (OA_TE * 100, AA_TE * 100, Kappa_TE))
    # test_end_time = time.time()
    # logger.info("Final:   Loss: %s  Accuracy: %s", test_loss, test_acc)
    #
    # train_time = train_end_time - train_start_time
    # test_time = test_end_time - test_start_time
    # # logger.debug('classification:\n %s\n confusion:\n%s\n ' % (classification, confusion))
    # logger.info("Train time:%s , Test time:%s", train_time, test_time)


def adjust_learning_rate(optimizer, epoch, learn_rate):
    lr = learn_rate * (0.1 ** (epoch // 50)) * (0.1 ** (epoch // 225))  # 每隔25个epoch更新学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()