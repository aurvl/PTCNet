import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ptck_arch.utils import ResidualConvBlock, MLPBlock


class MLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [MLPBlock(hidden_dim, hidden_dim * 2, dropout=dropout)
             for _ in range(n_blocks)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim)
        h = F.gelu(self.input_layer(x))   # proj en hidden_dim
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        out = self.output_layer(h)        # (B, 1)
        return out

class PTCNet(nn.Module):
    def __init__(
        self,
        n_features: int = 71,
        n_blocks: int = 10,
        kernel_size: int = 3,
        groups: int = 1,            # A: groups=n_features, B: groups=1
        dropout: float = 0.0,
        # --- pooling entre blocs ---
        pool_type: str = "none",    # "none", "max", "avg"
        pool_every: int = 1,        # applique pooling tous les 'pool_every' blocs
        pool_kernel: int = 2,
        pool_stride: int | None = None,
        # --- domain embedding / MLP ---
        n_domains: int | None = None,
        domain_emb_dim: int = 8,
        mlp_hidden_dim: int = 128,
        mlp_blocks: int = 4,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.pool_type = pool_type.lower()

        # 1) Blocs résiduels
        blocks = []
        for i in range(n_blocks):
            # decide si ce bloc fait conv+pool ou juste conv
            use_pool_here = (self.pool_type in ("max", "avg")) and ((i + 1) % pool_every == 0)
            block_pool_type = self.pool_type if use_pool_here else "none"

            blocks.append(
                ResidualConvBlock(
                    channels=n_features,
                    kernel_size=kernel_size,
                    groups=groups,
                    dropout=dropout,
                    pool_type=block_pool_type,
                    pool_kernel=pool_kernel,
                    pool_stride=pool_stride,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        # 2) pooling global final T->1 (on le laisse en avg pour l'instant)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 3) embedding de domaine (optionnel)
        self.n_domains = n_domains
        if n_domains is not None:
            self.domain_emb = nn.Embedding(n_domains, domain_emb_dim)
            mlp_input_dim = n_features + domain_emb_dim
        else:
            self.domain_emb = None
            mlp_input_dim = n_features

        # 4) MLP final -> 1 scalaire
        self.mlp = MLPHead(
            input_dim=mlp_input_dim,
            hidden_dim=mlp_hidden_dim,  # ex: 256 ou 512
            n_blocks=mlp_blocks,        # profondeur du head
            dropout=0.1,
        )

        # 5) tête de reconstruction pour le masked-step
        self.reconstruct_head = nn.Conv1d(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=1,
        )

    # ---------- backbone temporel brut ----------

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T)
        return self.blocks(x)  # T peut diminuer si pooling inter-bloc actif

    # ---------- encodage en 71 scalars ----------

    def encode_scalars(
        self,
        x: torch.Tensor,
        feature_mask: torch.Tensor | None = None,  # (B, F)
    ) -> torch.Tensor:
        """
        x: (B, F, T) -> s: (B, F)  (1 scalaire par feature)
        
        We use feature_mask to zero-out features that were padding in the original data.
        This ensures the model doesn't rely on padding values and improves interpretability.
        """
        h = self.forward_features(x)       # (B, F, T')
        h_pooled = self.global_pool(h)     # (B, F, 1)
        s = h_pooled.squeeze(-1)           # (B, F)

        if feature_mask is not None:
            # feature_mask is (B, F), s is (B, F)
            s = s * feature_mask

        return s

    # ---------- forward "normal" (score/logit) ----------

    def forward(
        self,
        x: torch.Tensor,
        feature_mask: torch.Tensor | None = None,
        domain_id: torch.Tensor | None = None,
        return_scalars: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Si return_scalars=False: retourne (B,)
        Si return_scalars=True : retourne (B,), (B, F)
        
        Note: time_masks (from collate) are not used here because padding is already zero
        and the Conv1d layers handle it naturally (or we assume it's negligible).
        Future attention-based models might need time_masks.
        """
        s = self.encode_scalars(x, feature_mask=feature_mask)  # (B, F)

        if self.domain_emb is not None and domain_id is not None:
            d = self.domain_emb(domain_id)  # (B, D)
            z = torch.cat([s, d], dim=-1)
        else:
            z = s

        out = self.mlp(z).squeeze(-1)  # (B,)

        if return_scalars:
            return out, s
        return out

    # ---------- self-supervised masked-step ----------

    def masked_step_loss(
        self,
        x: torch.Tensor,
        mask_prob: float = 0.15,
    ) -> torch.Tensor:
        """
        Self-supervised objective: Mask random time steps and try to reconstruct them.
        
        ⚠️ V1: suppose pas de pooling inter-bloc (pool_type='none').
        Si tu actives du pooling entre blocs, cette fonction ne sera plus cohérente
        pour reconstruire les time steps masqués.
        
        This function generates its OWN random time mask. It does NOT use the time_masks
        from the collate function. It assumes x is either unpadded or zero-padded.
        """
        if self.pool_type in ("max", "avg"):
            raise ValueError(
                "masked_step_loss V1 suppose pool_type='none' (pas de pooling entre blocs). "
                "Instancie un modèle sans pooling pour le pré-training self-supervised."
            )

        B, F, T = x.shape
        device = x.device

        x_orig = x.detach()

        time_mask = (torch.rand(B, 1, T, device=device) < mask_prob)
        x_masked = x.clone()
        x_masked = x_masked.masked_fill(time_mask.expand(-1, F, -1), 0.0)

        h = self.forward_features(x_masked)     # (B, F, T)
        x_rec = self.reconstruct_head(h)        # (B, F, T)

        full_mask = time_mask.expand(-1, F, -1)

        diff = (x_rec - x_orig) ** 2
        num_masked = full_mask.sum()

        if num_masked == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = (diff * full_mask).sum() / num_masked
        return loss

class ModelA_NoInteractions(PTCNet):
    """
    Modèle A : pas d'interactions entre features (depthwise conv).
    """
    def __init__(
        self,
        n_features: int = 71,
        n_blocks: int = 10,
        kernel_size: int = 3,
        dropout: float = 0.0,
        pool_type: str = "none",   # "none", "max", "avg"
        pool_every: int = 1,
        pool_kernel: int = 2,
        pool_stride: int | None = None,
        n_domains: int | None = None,
        domain_emb_dim: int = 8,
        mlp_hidden_dim: int = 128,
        mlp_blocks: int = 2,
    ):
        super().__init__(
            n_features=n_features,
            n_blocks=n_blocks,
            kernel_size=kernel_size,
            groups=n_features,  # depthwise
            dropout=dropout,
            pool_type=pool_type,
            pool_every=pool_every,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            n_domains=n_domains,
            domain_emb_dim=domain_emb_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_blocks=mlp_blocks,
        )


class ModelB_WithInteractions(PTCNet):
    """
    Modèle B : interactions entre features (conv full).
    """
    def __init__(
        self,
        n_features: int = 71,
        n_blocks: int = 10,
        kernel_size: int = 3,
        dropout: float = 0.0,
        pool_type: str = "none",   # "none", "max", "avg"
        pool_every: int = 1,
        pool_kernel: int = 2,
        pool_stride: int | None = None,
        n_domains: int | None = None,
        domain_emb_dim: int = 8,
        mlp_hidden_dim: int = 128,
        mlp_blocks: int = 2,
    ):
        super().__init__(
            n_features=n_features,
            n_blocks=n_blocks,
            kernel_size=kernel_size,
            groups=1,  # full conv
            dropout=dropout,
            pool_type=pool_type,
            pool_every=pool_every,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            n_domains=n_domains,
            domain_emb_dim=domain_emb_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_blocks=mlp_blocks,
        )

