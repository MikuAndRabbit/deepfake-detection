import torch.nn as nn


class PreNorm(nn.Module):
    """前置归一化层

    Args:
        dim (int): 归一化层输入向量的维度
        fn (function): 归一化后所要进行的计算
    """

    def __init__(self, dim: int, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """前馈神网络层

    Args:
        dim (int): 输入维度
        hidden_dim (int): 隐藏层维度
        dropout (float, optional): Dropout的概率. Defaults to 0..
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            # GELU为高斯误差线性单元
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
