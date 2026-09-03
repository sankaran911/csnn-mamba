# In[2]:
"""
FIXES APPLIED (see "# FIX" comments):
  1. class_info replaced with the real 15-class Houston 2013 (DFC2013)
     train/test split. The previous class_info was the Trento 6-class table
     with Houston-ish names slapped on -- totally mismatched with the 15
     class_names used in main_houston.py.
  2. test_coords is now derived using the SAME indices as hsi_test_samples /
     test_labels, guaranteeing the prediction/heat maps place predictions at
     the correct pixels. (Previously test_coords was built during the
     extraction pass, while the actual test arrays were rebuilt later with
     np.setdiff1d, which sorts indices -- the two lists could disagree.)
  3. test_loader now uses shuffle=False so predict() output order matches
     test_coords order exactly.
"""

import numpy as np
import torch.utils.data
import scipy.io as sio

from utils.auxiliary import applyPCA
from utils.hyper_pytorch import HyperData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path
# path='../Datasets/'
# C:/Users/admin/Desktop/Transformer_code/HSITransformer/Datasets/
houston_hsi = sio.loadmat('./data/Datasets/Houston2013/Houston.mat')['Houston']

print('HSI shape after PCA:', houston_hsi.shape)
houston_hsi = applyPCA(houston_hsi, 15)
print('houston_hsi shape:', houston_hsi.shape)

# Dummy LiDAR channel (Houston dataset has no LiDAR file)
Houston2013_lidar = np.zeros(
    (houston_hsi.shape[0], houston_hsi.shape[1], 1),
    dtype=np.float32
)

print('Dummy lidar shape:', Houston2013_lidar.shape)

#Load ground truth labels
gt_Houston2013 = sio.loadmat(
    './data/Datasets/Houston2013/Houston_gt.mat'
)['Houston_gt']
print('gt_Houston2013.shape:', gt_Houston2013.shape)


# # 2.0 Data Preprocessing & Dataloader Preparation

# 2.1 Define the class information
# FIX: real Houston 2013 (DFC2013) 15-class train/test split
# (previous table was the Trento 6-class table mislabeled with Houston names)
class_info = [
    (1,  "Healthy grass",    'training_sample', 198, 'test_sample', 1053, 'total', 1251),
    (2,  "Stressed grass",   'training_sample', 190, 'test_sample', 1064, 'total', 1254),
    (3,  "Synthetic grass",  'training_sample', 192, 'test_sample', 505,  'total', 697),
    (4,  "Trees",            'training_sample', 188, 'test_sample', 1056, 'total', 1244),
    (5,  "Soil",             'training_sample', 186, 'test_sample', 1056, 'total', 1242),
    (6,  "Water",            'training_sample', 182, 'test_sample', 143,  'total', 325),
    (7,  "Residential",      'training_sample', 196, 'test_sample', 1072, 'total', 1268),
    (8,  "Commercial",       'training_sample', 191, 'test_sample', 1053, 'total', 1244),
    (9,  "Road",             'training_sample', 193, 'test_sample', 1059, 'total', 1252),
    (10, "Highway",          'training_sample', 191, 'test_sample', 1036, 'total', 1227),
    (11, "Railway",          'training_sample', 181, 'test_sample', 1054, 'total', 1235),
    (12, "Parking Lot 1",    'training_sample', 192, 'test_sample', 1041, 'total', 1233),
    (13, "Parking Lot 2",    'training_sample', 184, 'test_sample', 285,  'total', 469),
    (14, "Tennis Court",     'training_sample', 181, 'test_sample', 247,  'total', 428),
    (15, "Running Track",    'training_sample', 187, 'test_sample', 473,  'total', 660),
]

# Create a dictionary to store class number, class name, and class samples
class_dict = {class_number: {"class_name": class_name,
                             'training_sample': training_sample,
                             'test_sample': test_sample,
                             "total_samples": total}
              for class_number, class_name, _, training_sample, _, test_sample, _, total in class_info}

print(class_dict)
print("\nTable 2: Houston 2013 Dataset")
print("-" * 75)
print("{:<5} {:<20} {:<10} {:<10} {:<10}".format(
    "No.", "Class Name", "Training", "Test", "Samples"))

total_train = 0
total_test = 0
total_samples = 0

for item in class_info:
    no = item[0]
    name = item[1]
    train = item[3]
    test = item[5]
    samples = item[7]

    print("{:<5} {:<20} {:<10} {:<10} {:<10}".format(
        no, name, train, test, samples))

    total_train += train
    total_test += test
    total_samples += samples

print("-" * 75)
print("{:<5} {:<20} {:<10} {:<10} {:<10}".format(
    "", "Total", total_train, total_test, total_samples))


# ### 2.1  Samples Extraction

# 2.2 Samples Extraction

# Define patch size and stride
patch_size = 21
stride = 1

# Create an empty list to store patches and labels
hsi_samples = []
lidar_samples = []
labels = []
coords_samples = []  # FIX: track (row, col) of every accepted patch, in lockstep

# Initialize a dictionary to store class count
class_count = {i: 0 for i in class_dict.keys()}

# Function to check if all classes have the required number of samples
def all_classes_completed(class_count, class_dict):
    return all(class_count[class_num] == class_dict[class_num]["total_samples"] for class_num in class_dict.keys())

while not all_classes_completed(class_count, class_dict):
    # Loop through the ground truth data
    for label in class_dict.keys():
        # Get the coordinates of the ground truth pixels
        #coords = np.argwhere((gt_2013_data == label) & (mask > 0))
        coords = np.argwhere(gt_Houston2013 == label)

        # Shuffle the coordinates to randomize the patch extraction
        np.random.shuffle(coords)

        for coord in coords:
            i, j = coord
            # Calculate the patch indices
            i_start, i_end = i - patch_size // 2, i + patch_size // 2 + 1
            j_start, j_end = j - patch_size // 2, j + patch_size // 2 + 1

            # Check if the indices are within the bounds of the HSI data
            if i_start >= 0 and i_end <= houston_hsi.shape[0] and j_start >= 0 and j_end <= houston_hsi.shape[1]:
                # Extract the patch
                hsi_patch = houston_hsi[i_start:i_end, j_start:j_end, :]

                # Extract the LiDAR patch
                lidar_patch = Houston2013_lidar[i_start:i_end, j_start:j_end, :]

                # If the class count is less than the required samples
                if class_count[label] < class_dict[label]["total_samples"]:
                    # Append the patch and its label to the list
                    hsi_samples.append(hsi_patch)
                    lidar_samples.append(lidar_patch)
                    labels.append(label)
                    coords_samples.append((i, j))  # FIX: record coord for this exact patch

                    class_count[label] += 1
                    # If all classes have the required number of samples, exit the loop
                    if all_classes_completed(class_count, class_dict):
                        break

# Convert the list of patches and labels into arrays
hsi_samples = np.array(hsi_samples)
lidar_samples = np.array(lidar_samples)
labels = np.array(labels) # GT
coords_samples = np.array(coords_samples)  # FIX: aligned 1:1 with hsi_samples/labels
print('hsi_samples shape:', hsi_samples.shape)
print('lidar_samples shape:', lidar_samples.shape)
print('labels shape:', labels.shape)


# ### 2.2 Training samples extraction

#Avoid overlap of train and test
# Extracting training samples
hsi_training_samples, lidar_training_samples, training_labels = [], [], []
used_indices = []  # To keep track of indices already taken for training samples

for label, class_data in class_dict.items():
    # Get indices of the current class
    class_indices = np.where(labels == label)[0]

    # Randomly shuffle the indices
    np.random.shuffle(class_indices)

    # Take the required number of training samples
    train_indices = class_indices[:class_data["training_sample"]]
    used_indices.extend(train_indices)  # Add these to the used_indices list

    # Append training samples
    hsi_training_samples.extend(hsi_samples[train_indices])
    lidar_training_samples.extend(lidar_samples[train_indices])
    training_labels.extend(labels[train_indices])

# Extracting test samples
hsi_test_samples, lidar_test_samples, test_labels = [], [], []
test_coords = []  # FIX: rebuilt here, using the exact same indices as the test arrays below

for label, class_data in class_dict.items():
    class_indices = np.where(labels == label)[0]

    # Exclude indices which were used for training
    test_indices = np.setdiff1d(class_indices, used_indices)

    # Append test samples
    hsi_test_samples.extend(hsi_samples[test_indices])
    lidar_test_samples.extend(lidar_samples[test_indices])
    test_labels.extend(labels[test_indices])
    test_coords.extend(coords_samples[test_indices])  # FIX: same indices -> guaranteed alignment

# Convert lists back to numpy arrays
hsi_training_samples = np.array(hsi_training_samples)
lidar_training_samples = np.array(lidar_training_samples)
training_labels = np.array(training_labels)

hsi_test_samples = np.array(hsi_test_samples)
lidar_test_samples = np.array(lidar_test_samples)
test_labels = np.array(test_labels)
test_coords = np.array(test_coords)  # FIX: now safe to rely on this in main_houston.py

# FIX: shift labels from 1-15 down to 0-14. CrossEntropyLoss requires targets
# in [0, num_classes-1]; class_dict uses 1-15, so label "15" was out of range
# for a 15-unit output layer -> CUDA assertion `t < n_classes` failed.
# (main_houston.py already assumes 0-indexed predictions and does
# `pred_labels = argmax(...) + 1` to map back to the original 1-15 numbering,
# so this makes that assumption actually true.)
training_labels = training_labels - 1
test_labels = test_labels - 1

# Print shapes to verify
print('hsi_training_samples shape:', hsi_training_samples.shape)
print('lidar_training_samples shape:', lidar_training_samples.shape)
print('training_labels shape:', training_labels.shape)

print('hsi_test_samples shape:', hsi_test_samples.shape)
print('lidar_test_samples shape:', lidar_test_samples.shape)
print('test_labels shape:', test_labels.shape)
print('test_coords shape:', test_coords.shape)

# hsi_train=np.transpose(hsi_training_samples, (0, 3, 1, 2))
hsi_train=hsi_training_samples
lidar_train=lidar_training_samples
y_train=training_labels
print('hsi_train_samples shape:', hsi_train.shape)
print('lidar_train_samples shape:', lidar_train.shape)
print('train_labels shape:', y_train.shape)
hsi_test=hsi_test_samples
lidar_test=lidar_test_samples
y_test=test_labels
print('hsi_test_samples shape:', hsi_test.shape)
print('lidar_test_samples shape:', lidar_test.shape)
print('y_test shape:', y_test.shape)


train_multimodal = np.concatenate((hsi_training_samples,lidar_training_samples),axis=3)
test_multimodal = np.concatenate((hsi_test_samples,lidar_test_samples), axis=3)

train_multimodal = HyperData((np.transpose(train_multimodal, (0, 3, 1, 2)).astype("float32"), training_labels),None)
test_multimodal = HyperData((np.transpose(test_multimodal, (0, 3, 1, 2)).astype("float32"), test_labels),None)


train_loader = torch.utils.data.DataLoader(dataset=train_multimodal,batch_size=32,
                                               shuffle=True,
                                               num_workers=0)
# FIX: shuffle=False -- prediction order must match test_coords order 1:1
# for the prediction map / heat map in main_houston.py to place labels at the
# right pixels.
test_loader = torch.utils.data.DataLoader(dataset=test_multimodal,batch_size=32,
                                               shuffle=False,
                                               num_workers=0)
print("data is ok")

print("Test coordinates:", len(test_coords))