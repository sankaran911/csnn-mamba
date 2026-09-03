import scipy.io as sio

mat = sio.loadmat("./data/Datasets/muufl/MUUFL_TruthForSubImage.mat")

gt = mat["MUUFL_Gulfport_GroundTruth"]

print(type(gt))
print(gt.shape)

print("\nField names:")
print(gt.dtype.names)