import torch
import torch.nn as nn

class SimpleMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv1d = nn.Conv1d(
            dim,
            dim,
            kernel_size=3,
            padding=1
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        T,B,C,H,W = x.shape

        x = x.permute(1,3,4,0,2)
        x = x.reshape(B*H*W,T,C)

        x = x.transpose(1,2)

        x = self.conv1d(x)

        x = x.transpose(1,2)

        x = self.norm(x)

        x = x.reshape(B,H,W,T,C)

        x = x.permute(3,0,4,1,2)

        return x