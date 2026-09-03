import scipy.io as sio
import matplotlib.pyplot as plt

# Load Ground Truth
gt = sio.loadmat('./data/Datasets/Trento/GT_Trento.mat')['GT_Trento']

plt.figure(figsize=(8,6))
plt.imshow(gt, cmap='tab20')
plt.title("Trento Ground Truth")
plt.axis('off')

plt.savefig("Trento_Ground_Truth.png", dpi=300, bbox_inches='tight')
plt.show()