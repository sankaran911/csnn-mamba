# -*- coding: utf-8 -*-
"""
@CreatedDate:   2020/4/27 12:08
@Author: Pangpd(https://github.com/pangpd/DS-pResNet-HSI)
@UsedBy: lyh

ADDITIONS (see "# FIX" / "# NEW" comments):
  1. import dataload_houston  -- needed to reach gt_Houston2013 / test_coords,
     which were referenced nowhere before but are required for the maps below.
  2. Added: Ground Truth map, Confusion Matrix, discrete-class Prediction Map,
     and a softmax-confidence Heat Map -- none of these existed before.
  3. predict() was never called in the original file -- added it.
"""
import os
import sys
import time
import numpy as np

import dataload_houston  # FIX: needed for dataload_houston.gt_Houston2013 / test_coords
from dataload_houston import train_loader, test_loader
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
#from qs_mamba_houston import ComplexNet as SNN
from complexNet_Houston2013 import ComplexNet as SNN
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

dataset = 'HOUSTON'  # 数据集
seed = 1014
epochs = 50

learn_rate = 0.0085
# learn_rate = 0.00001
momentum = 0.9
weight_decay = 0.0001
class_number = 15

iter = 1

class_names = [
    "Healthy grass",
    "Stressed grass",
    "Synthetic grass",
    "Trees",
    "Soil",
    "Water",
    "Residential",
    "Commercial",
    "Road",
    "Highway",
    "Railway",
    "Parking Lot 1",
    "Parking Lot 2",
    "Tennis Court",
    "Running Track"
]

# NEW: one color per class (index 0 = background/unlabeled), used for both
# the ground truth map and the prediction map so they're visually comparable.
CLASS_COLORS = np.array([
    [0, 0, 0],        # 0  background
    [0, 205, 0],      # 1  Healthy grass
    [127, 255, 0],    # 2  Stressed grass
    [46, 139, 87],    # 3  Synthetic grass
    [0, 100, 0],      # 4  Trees
    [160, 82, 45],    # 5  Soil
    [0, 255, 255],    # 6  Water
    [255, 255, 0],    # 7  Residential
    [255, 165, 0],    # 8  Commercial
    [255, 0, 0],      # 9  Road
    [139, 0, 0],      # 10 Highway
    [128, 0, 128],    # 11 Railway
    [255, 192, 203],  # 12 Parking Lot 1
    [255, 105, 180],  # 13 Parking Lot 2
    [0, 0, 255],      # 14 Tennis Court
    [30, 144, 255],   # 15 Running Track
], dtype=np.uint8)


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

    import matplotlib.pyplot as plt

    # NEW: Ground Truth map -- doesn't depend on the model, so save it once
    # up front.
    gt_rgb = CLASS_COLORS[dataload_houston.gt_Houston2013]
    plt.figure(figsize=(12, 5))
    plt.imshow(gt_rgb)
    plt.title("Houston 2013 Ground Truth")
    plt.axis("off")
    plt.savefig(os.path.join(group_log_dir, "Houston_Ground_Truth.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # model = SNN(10, leak_mem=0.7, img_size=spatial_size, num_cls=class_number, input_dim=components) #SA最好精度在40个步长
    model = SNN(10, input_dim=15)

     #print(model)
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
        valid_loss, valid_acc , test_acc1, test_obj, tar_v, pre_v= test(test_loader, model, criterion, epoch, use_cuda)
        print("AFTER TEST()")
        print("tar_v length =", len(tar_v))
        print("pre_v length =", len(pre_v))
        print("HELLO AFTER TEST")

        print("SKIPPED LOGGER")

        print("BEFORE OUTPUT_METRIC")
        OA_TE, AA_TE, Kappa_TE, CA_TE = output_metric(tar_v, pre_v)
        print("AFTER OUTPUT_METRIC")

        print("================================")
        print("OA =", OA_TE * 100)
        print("AA =", AA_TE * 100)
        print("Kappa =", Kappa_TE)

        for cls, acc in zip(class_names, CA_TE * 100):
            print(f"{cls}: {acc:.2f}%")

        print("================================")

        # NEW: Confusion Matrix (only on the last epoch, to avoid writing
        # 50 images -- move this outside the `if epoch == epochs - 1` check
        # if you want one per epoch instead).
        if epoch == epochs - 1:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(tar_v, pre_v)

            plt.figure(figsize=(12, 10))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Houston 2013 Confusion Matrix")
            plt.colorbar()

            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=90)
            plt.yticks(tick_marks, class_names)

            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(
                        j, i, str(cm[i, j]),
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if cm[i, j] > cm.max() / 2 else "black"
                    )

            plt.tight_layout()
            plt.savefig(os.path.join(group_log_dir, "Houston_Confusion_Matrix.png"), dpi=300)
            plt.close()

        # save model
        if valid_acc > best_acc:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': valid_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, group_log_dir + "/best_model.pth_Houston.tar")
            best_acc = valid_acc
            best_oa = OA_TE * 100
            best_aa = AA_TE * 100
            best_kappa = Kappa_TE
            best_each_acc = CA_TE * 100
    if logger:
     logger.info('best_AA: %f, best_OA: %f, best_kappa: %f\n ' % (best_aa, best_oa, best_kappa))
     logger.info('best_CA: %s \n', best_each_acc)

    # ================= Prediction / Prediction Map / Heat Map =================
    # NEW: predict() was never called in the original file.
    pred = predict(test_loader, model, use_cuda)  # expected shape: (N, num_classes)
    print("Prediction shape:", pred.shape)
    print("First prediction:", pred[0])

    pred_tensor = torch.from_numpy(pred) if isinstance(pred, np.ndarray) else pred
    pred_probs = F.softmax(pred_tensor, dim=1).cpu().numpy()

    pred_labels = np.argmax(pred_probs, axis=1) + 1   # +1 to reserve 0 for background
    confidence = np.max(pred_probs, axis=1)           # max softmax prob per pixel
    print("Predicted labels shape:", pred_labels.shape)

    # ---------- Discrete class Prediction Map ----------
    prediction_map = np.zeros(dataload_houston.gt_Houston2013.shape, dtype=np.uint8)
    for (r, c), label in zip(dataload_houston.test_coords, pred_labels):
        prediction_map[r, c] = label

    rgb = CLASS_COLORS[prediction_map]

    plt.figure(figsize=(12, 5))
    plt.imshow(rgb)
    plt.title("Houston 2013 Prediction Map")
    plt.axis("off")
    plt.savefig(os.path.join(group_log_dir, "Houston_Prediction.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ---------- Confidence Heat Map ----------
    confidence_map = np.zeros(dataload_houston.gt_Houston2013.shape, dtype=np.float32)
    for (r, c), conf in zip(dataload_houston.test_coords, confidence):
        confidence_map[r, c] = conf

    masked_confidence = np.ma.masked_where(confidence_map == 0, confidence_map)

    plt.figure(figsize=(12, 5))
    cmap = plt.cm.jet
    cmap.set_bad(color="black")
    im = plt.imshow(masked_confidence, cmap=cmap, vmin=0, vmax=1)
    plt.title("Houston 2013 Prediction Confidence Heat Map")
    plt.axis("off")
    cbar = plt.colorbar(im, fraction=0.03, pad=0.02)
    cbar.set_label("Softmax Confidence")
    plt.savefig(os.path.join(group_log_dir, "Houston_Heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()

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