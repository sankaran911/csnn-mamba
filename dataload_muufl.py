"""
dataload_muufl.py
-----------------
Data loading and preprocessing for the MUUFL Gulfport dataset.

Uses the OFFICIAL scene-labeled benchmark file from GatorSense:
    muufl_gulfport_campus_1_hsi_220_label.mat
    (https://github.com/GatorSense/MUUFLGulfport/tree/master/MUUFLGulfportSceneLabels)

This file (NOT MUUFL_TruthForSubImage.mat) contains a full dense per-pixel
ground truth map under hsi.sceneLabels.labels, covering 11 land-cover classes
across the whole 325 x 220 scene -- this is the standard benchmark used in
HSI+LiDAR joint-classification papers.

Struct layout (loaded with struct_as_record=False, squeeze_me=True):
    hsi.Data                      -> (325, 220, 64)  HSI cube
    hsi.Lidar[0].z                -> (325, 220)      LiDAR elevation (DEM)
    hsi.Lidar[1].z                -> (325, 220)      LiDAR elevation (2nd return), if present
    hsi.sceneLabels.labels        -> (325, 220)      dense GT map, -1/0 = unlabeled, 1-11 = classes
    hsi.sceneLabels.Materials_Type-> list of 11 class-name strings
"""

import numpy as np
import scipy.io as sio
import torch
import torch.utils.data

from utils.auxiliary import applyPCA
from utils.hyper_pytorch import HyperData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load the OFFICIAL scene-labeled MUUFL file
# ─────────────────────────────────────────────────────────────────────────────
MUUFL_LABEL_MAT = "./data/Datasets/muufl/muufl_gulfport_campus_1_hsi_220_label.mat"

d = sio.loadmat(
    MUUFL_LABEL_MAT,
    squeeze_me=True,
    mat_dtype=True,
    struct_as_record=False,
)["hsi"]

# ── HSI cube ──────────────────────────────────────────────────────────────────
muufl_hsi_raw = np.asarray(d.Data, dtype=np.float32)        # (325, 220, 64)
print("HSI raw shape:", muufl_hsi_raw.shape)

# Apply PCA -> 15 components (consistent with Houston / Trento pipelines)
muufl_hsi = applyPCA(muufl_hsi_raw, 15)
print("HSI shape after PCA:", muufl_hsi.shape)               # (325, 220, 15)

# ── LiDAR ─────────────────────────────────────────────────────────────────────
# d.Lidar is an array of Lidar-return structs; each has a .z (elevation) field.
# --- LiDAR ---
lidar = np.asarray(d.Lidar[0].z, dtype=np.float32)

def norm01(arr):
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

for c in range(lidar.shape[2]):
    lidar[:, :, c] = norm01(lidar[:, :, c])

muufl_lidar = lidar

LIDAR_CHANNELS = muufl_lidar.shape[2]

print("LiDAR shape:", muufl_lidar.shape)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Ground truth -- dense (325, 220) scene label map
# ─────────────────────────────────────────────────────────────────────────────
gt_MUUFL = np.asarray(d.sceneLabels.labels, dtype=np.int64)  # (325, 220)
gt_MUUFL[gt_MUUFL < 0] = 0      # unlabeled / background -> 0

print("Ground Truth shape:", gt_MUUFL.shape)
print("Unique labels:     ", np.unique(gt_MUUFL))
print("Labelled pixels:   ", np.count_nonzero(gt_MUUFL))

class_names_from_file = list(d.sceneLabels.Materials_Type)
print("Class names from file:", class_names_from_file)

NUM_CLASSES = len(class_names_from_file)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Class information  -- standard MUUFL benchmark protocol of 100
#    training samples per class (clamped if a class has fewer pixels).
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_PER_CLASS = 100  # matches common MUUFL benchmark protocol (100 train/class)

class_dict = {}
for cls_num in range(1, NUM_CLASSES + 1):
    actual_total = int(np.sum(gt_MUUFL == cls_num))
    n_train = min(TRAIN_PER_CLASS, max(1, actual_total // 2))
    n_test = actual_total - n_train

    class_dict[cls_num] = {
        "class_name": class_names_from_file[cls_num - 1],
        "training_sample": n_train,
        "test_sample": n_test,
        "total_samples": actual_total,
    }

print("\nTable: MUUFL Gulfport Dataset (official scene labels)")
print("-" * 75)
print("{:<5} {:<25} {:<10} {:<10} {:<10}".format(
    "No.", "Class Name", "Training", "Test", "Samples"))
for cls_num, info in class_dict.items():
    print("{:<5} {:<25} {:<10} {:<10} {:<10}".format(
        cls_num, info["class_name"],
        info["training_sample"], info["test_sample"], info["total_samples"]))
print("-" * 75)
print("{:<5} {:<25} {:<10} {:<10} {:<10}".format(
    "", "Total",
    sum(v["training_sample"] for v in class_dict.values()),
    sum(v["test_sample"] for v in class_dict.values()),
    sum(v["total_samples"] for v in class_dict.values())))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Patch extraction
# ─────────────────────────────────────────────────────────────────────────────
patch_size = 9

hsi_samples = []
lidar_samples = []
sample_labels = []
sample_coords = []

class_count = {i: 0 for i in class_dict.keys()}


def all_classes_completed(cc, cd):
    return all(cc[k] >= cd[k]["total_samples"] for k in cd.keys())


while not all_classes_completed(class_count, class_dict):
    progressed = False
    for label in class_dict.keys():
        needed = class_dict[label]["total_samples"] - class_count[label]
        if needed <= 0:
            continue

        coords = np.argwhere(gt_MUUFL == label)
        #np.random.shuffle(coords)

        for coord in coords:
            i, j = coord
            i_start = i - patch_size // 2
            i_end = i + patch_size // 2 + 1
            j_start = j - patch_size // 2
            j_end = j + patch_size // 2 + 1

            if (i_start >= 0 and i_end <= muufl_hsi.shape[0] and
                    j_start >= 0 and j_end <= muufl_hsi.shape[1]):

                if class_count[label] < class_dict[label]["total_samples"]:
                    hsi_samples.append(muufl_hsi[i_start:i_end, j_start:j_end, :])
                    lidar_samples.append(muufl_lidar[i_start:i_end, j_start:j_end, :])
                    sample_labels.append(label)
                    sample_coords.append((i,j))
                    class_count[label] += 1
                    progressed = True

                    if all_classes_completed(class_count, class_dict):
                        break
    # Safety valve: if a class can never reach its target (e.g. too close to
    # image border for every remaining pixel), shrink its target instead of
    # looping forever.
    if not progressed:
        for label in class_dict.keys():
            if class_count[label] < class_dict[label]["total_samples"]:
                class_dict[label]["total_samples"] = class_count[label]
                if class_dict[label]["training_sample"] > class_count[label]:
                    class_dict[label]["training_sample"] = max(1, class_count[label] // 2)
                class_dict[label]["test_sample"] = (
                    class_count[label] - class_dict[label]["training_sample"])

hsi_samples = np.array(hsi_samples, dtype=np.float32)
lidar_samples = np.array(lidar_samples, dtype=np.float32)
sample_labels = np.array(sample_labels, dtype=np.int64)

sample_coords = np.array(sample_coords)

print("\nhsi_samples shape:   ", hsi_samples.shape)
print("lidar_samples shape: ", lidar_samples.shape)
print("labels shape:        ", sample_labels.shape)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Train / test split  (no overlap)
# ─────────────────────────────────────────────────────────────────────────────
hsi_training_samples = []
lidar_training_samples = []
training_labels = []
used_indices = []

for label, info in class_dict.items():
    cls_indices = np.where(sample_labels == label)[0]
    np.random.shuffle(cls_indices)

    train_idx = cls_indices[:info["training_sample"]]
    used_indices.extend(train_idx.tolist())

    hsi_training_samples.extend(hsi_samples[train_idx])
    lidar_training_samples.extend(lidar_samples[train_idx])
    training_labels.extend(sample_labels[train_idx])

hsi_test_samples = []
lidar_test_samples = []
test_labels = []
test_indices = []
test_coords = []

for label, info in class_dict.items():
    cls_indices = np.where(sample_labels == label)[0]
    test_idx = np.setdiff1d(cls_indices, used_indices)

    test_indices.extend(test_idx.tolist())

    hsi_test_samples.extend(hsi_samples[test_idx])
    lidar_test_samples.extend(lidar_samples[test_idx])
    test_labels.extend(sample_labels[test_idx])

    # ADD THIS LINE
    test_coords.extend(sample_coords[test_idx])

hsi_training_samples = np.array(hsi_training_samples, dtype=np.float32)
lidar_training_samples = np.array(lidar_training_samples, dtype=np.float32)
training_labels = np.array(training_labels, dtype=np.int64) - 1

hsi_test_samples = np.array(hsi_test_samples, dtype=np.float32)
lidar_test_samples = np.array(lidar_test_samples, dtype=np.float32)
test_labels = np.array(test_labels, dtype=np.int64) - 1
test_coords = np.array(test_coords, dtype=np.int64)

print("\nhsi_training_samples shape:   ", hsi_training_samples.shape)
print("lidar_training_samples shape: ", lidar_training_samples.shape)
print("training_labels shape:        ", training_labels.shape)
print("hsi_test_samples shape:       ", hsi_test_samples.shape)
print("lidar_test_samples shape:     ", lidar_test_samples.shape)
print("test_labels shape:            ", test_labels.shape)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Build DataLoaders
#    Channel layout: first 15 = HSI (PCA),  next LIDAR_CHANNELS = LiDAR
# ─────────────────────────────────────────────────────────────────────────────
print("HSI training:", hsi_training_samples.shape)
print("LiDAR training:", lidar_training_samples.shape)
print("HSI test:", hsi_test_samples.shape)
print("LiDAR test:", lidar_test_samples.shape)

train_multimodal = np.concatenate(
    (hsi_training_samples, lidar_training_samples), axis=3)   # (N, 15, 15, 15+L)
test_multimodal = np.concatenate(
    (hsi_test_samples, lidar_test_samples), axis=3)

# Transpose to (N, C, H, W)
train_multimodal = HyperData(
    (np.transpose(train_multimodal, (0, 3, 1, 2)), training_labels), None)
test_multimodal = HyperData(
    (np.transpose(test_multimodal, (0, 3, 1, 2)), test_labels), None)

train_loader = torch.utils.data.DataLoader(
    dataset=train_multimodal, batch_size=32, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(
    dataset=test_multimodal, batch_size=32, shuffle=False, num_workers=0)

print("\ndata is ok")
print(f"train batches: {len(train_loader)} | test batches: {len(test_loader)}")
print(f"NUM_CLASSES = {NUM_CLASSES} | LIDAR_CHANNELS = {LIDAR_CHANNELS}")

from show_maps import show_gt

show_gt(
    data=muufl_hsi,
    labels=gt_MUUFL,
    class_nums=NUM_CLASSES,
    save_path="ground_truth.png",
    class_names=class_names_from_file
)