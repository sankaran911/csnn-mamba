import torch
from mamba_block import SimpleMamba

# Same dimensions used by your CSNN
T = 10
B = 2
C = 64
H = 21
W = 21

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# Simulated CSNN/LIF output
x = torch.randn(T, B, C, H, W).to(device)

print("Input shape :", x.shape)

mamba = SimpleMamba(C).to(device)

y = mamba(x)

print("Output shape:", y.shape)

# Verify
assert y.shape == x.shape

print("SimpleMamba test PASSED")
print("Input and output shapes are identical.")