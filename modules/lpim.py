import torch
import torch.nn as nn


class LPIM(nn.Module):
    """
    Learnable Patch Importance Module

    Input
    -----
    patch_tokens : (B, 256, 1280)

    Output
    ------
    selected_tokens : (B, top_k, 1280)

    selected_indices : (B, top_k)

    scores : (B, 256)
    """

    def __init__(
        self,
        embed_dim=1280,
        hidden_dim=512,
        top_k=64,
        dropout=0.1,
    ):
        super().__init__()

        self.top_k = top_k

        self.scorer = nn.Sequential(

            nn.Linear(embed_dim, hidden_dim),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1),

        )

    def forward(self, patch_tokens):

        scores = self.scorer(patch_tokens).squeeze(-1)

        _, indices = torch.topk(
            scores,
            k=self.top_k,
            dim=1,
        )

        selected_tokens = torch.gather(
            patch_tokens,
            dim=1,
            index=indices.unsqueeze(-1).expand(
                -1,
                -1,
                patch_tokens.size(-1),
            ),
        )

        return selected_tokens, indices, scores
