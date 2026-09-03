# In[2]:
"""
FIXES APPLIED (see "# FIX" comments):
  1. class_count[label] += 1 was nested inside the training-threshold check,
     so it could never increment -> infinite while loop. Moved it up so it
     increments on every accepted patch, regardless of train/test phase.
  2. class_info names corrected from Houston-style labels ("Healthy grass"...)
     to the actual Trento classes. The sample-count numbers were already
     correct for Trento, only the names were wrong.
  3. test_coords is now derived in the SAME loop, using the SAME indices, as
     hsi_test_samples / test_labels -> guarantees the prediction map lines up
     with the right pixels. (Previously test_coords was built during the
     extraction pass while the actual test arrays were rebuilt later with
     np.setdiff1d, which sorts indices -> the two lists could disagree.)
  4. test_loader now uses shuffle=False, so the order predict() returns
     results in matches test_coords exactly. (train_loader shuffling is
     fine since training order doesn't need to map back to pixels.)
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

# 2.1 Loads Data
# Load hyperpsectral data
trento_hsi=sio.loadmat('./data/Datasets/Trento/HSI_Trento.mat')['HSI_Trento']
print('trento_hsi shape:', trento_hsi.shape)
trento_hsi = applyPCA(trento_hsi, 15)
print("HSI shape after PCA:", trento_hsi.shape)

# Loader Lidar  data
trento_lidar = sio.loadmat('./data/Datasets/Trento/Lidar1_Trento.mat')['Lidar_Trento']
trento_lidar = np.expand_dims(trento_lidar,2)
print('trento_lidar shape:', trento_lidar.shape)

#Load ground truth labels
gt_trento=sio.loadmat('./data/Datasets/Trento/GT_Trento.mat')['GT_Trento']
print('gt_trento.shape:', gt_trento.shape)


# # 2.0 Data Preprocessing & Dataloader Preparation

# 2.1 Define the class information
# FIX: corrected class names to match the actual Trento dataset
# (counts were already correct: 4034/2903/479/9123/10501/3174 total pixels)
class_info = [(1, "Apple Trees",   'training_sample', 129, 'test_sample', 3905,  'total', 4034),
    (2, "Buildings",   'training_sample', 125, 'test_sample', 278,  'total', 2903),
    (3, "Ground",      'training_sample', 105, 'test_sample', 374,  'total', 479),
    (4, "Wood",        'training_sample', 154, 'test_sample', 8969,  'total', 9123),
    (5, "Vineyard",    'training_sample', 184, 'test_sample', 10317,  'total', 10501),
    (6, "Roads",       'training_sample', 122, 'test_sample', 3052,  'total', 3174)]

# Create a dictionary to store class number, class name, and class samples
class_dict = {class_number: {"class_name": class_name,
                             'training_sample': training_sample,
                             'test_sample': test_sample,
                             "total_samples": total}
              for class_number, class_name, _, training_sample, _, test_sample, _, total in class_info}

print(class_dict)


# ### 2.1  Samples Extraction

# 2.2 Samples Extraction

# Define patch size and stride
patch_size = 13
stride = 1

# Create an empty list to store patches and labels
hsi_samples = []
lidar_samples = []
labels = []
coords_samples = []  # FIX: track the (row, col) of every accepted patch, in lockstep

# Initialize a dictionary to store class count
class_count = {i: 0 for i in class_dict.keys()}

# Function to check if all classes have the required number of samples
def all_classes_completed(class_count, class_dict):
    return all(class_count[class_num] == class_dict[class_num]["total_samples"] for class_num in class_dict.keys())

#print("Starting patch extraction...")
#iteration = 0

while not all_classes_completed(class_count, class_dict):
    #iteration += 1
    #print("Iteration:", iteration)
    #print(class_count)
    #print("______________________")
    # Loop through the ground truth data
    for label in class_dict.keys():
        # Get the coordinates of the ground truth pixels
        #coords = np.argwhere((gt_2013_data == label) & (mask > 0))
        coords = np.argwhere(gt_trento == label)

        # Shuffle the coordinates to randomize the patch extraction
        np.random.shuffle(coords)

        for coord in coords:
            #print(i,j)
            #print(i_start,i_end,j_start,j_end)
            i, j = coord
            # Calculate the patch indices
            i_start, i_end = i - patch_size // 2, i + patch_size // 2 + 1
            j_start, j_end = j - patch_size // 2, j + patch_size // 2 + 1

            # Check if the indices are within the bounds of the HSI data
            if i_start >= 0 and i_end <= trento_hsi.shape[0] and j_start >= 0 and j_end <= trento_hsi.shape[1]:
                # Extract the patch
                hsi_patch = trento_hsi[i_start:i_end, j_start:j_end, :]

                # Extract the LiDAR patch
                lidar_patch = trento_lidar[i_start:i_end, j_start:j_end, :]

                # If the class count is less than the required samples
                if class_count[label] < class_dict[label]["total_samples"]:
                    # Append the patch and its label to the list
                    hsi_samples.append(hsi_patch)
                    lidar_samples.append(lidar_patch)
                    labels.append(label)
                    coords_samples.append((i, j))  # FIX: record coord for this exact patch

                    # FIX: increment on every accepted patch (was nested one level
                    # too deep before, so it could never fire -> infinite loop)
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
test_coords = np.array(test_coords)  # FIX: now safe to rely on this in main_trento.py

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
# for the prediction map / heat map in main_trento.py to place labels at the
# right pixels.
test_loader = torch.utils.data.DataLoader(dataset=test_multimodal,batch_size=32,
                                               shuffle=False,
                                               num_workers=0)
print("data is ok")


print("Test coordinates:", len(test_coords))