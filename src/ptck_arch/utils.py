import torch
import torch.nn as nn
import torch.nn.functional as F


class SafePool1d(nn.Module):
    """
    Pooling 1D qui:
      - applique MaxPool ou AvgPool quand la longueur L >= kernel_size
      - sinon renvoie x tel quel (pour ne jamais descendre en dessous de 1).
    """
    def __init__(self, mode: str = "max", kernel_size: int = 2, stride: int | None = None):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride

        mode = mode.lower()
        if mode == "max":
            self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)
        elif mode == "avg":
            self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)
        else:
            raise ValueError(f"Unknown pool mode {mode!r}, use 'max' or 'avg'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        if x.size(-1) >= self.kernel_size:
            return self.pool(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        groups: int = 1,          # groups=channels -> depthwise, 1 -> full conv
        dropout: float = 0.0,
        pool_type: str = "none",  # "none", "max", "avg"
        pool_kernel: int = 2,
        pool_stride: int | None = None,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # projection identité (au cas où on change les channels plus tard)
        self.proj = nn.Identity()

        pool_type = pool_type.lower()
        if pool_type in ("max", "avg"):
            self.pool = SafePool1d(
                mode=pool_type,
                kernel_size=pool_kernel,
                stride=pool_stride,
            )
        else:
            self.pool = None  # pas de pooling inter-bloc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        identity = self.proj(x)

        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)

        # Si on a un pooling inter-bloc, on l'applique
        if self.pool is not None:
            out = self.pool(out)
            # Il faut aussi downsampler l'identity pour matcher T
            identity = self.pool(identity)

        return out + identity


class MLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lin1 = nn.Linear(dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, dim)
        residual = x
        out = self.norm(x)
        out = F.gelu(self.lin1(out))
        out = self.dropout(out)
        out = self.lin2(out)
        out = self.dropout(out)
        return residual + out