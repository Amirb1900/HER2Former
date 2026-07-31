# کل کد TopKSelector اینجا
import torch
import torch.nn as nn


class TopKSelector(nn.Module):
    """
    Select Top-K informative patch tokens.

    Input:
        patch_tokens : (B, N, C)

    Output:
        selected_tokens : (B, K, C)
        scores          : (B, N)
        indices         : (B, K)
    """

    def __init__(self, embed_dim=1280, top_k=64):
        super().__init__()

        self.top_k = top_k

        self.score_layer = nn.Linear(embed_dim, 1)

    def forward(self, patch_tokens):

        # (B,N)
        scores = self.score_layer(
            patch_tokens
        ).squeeze(-1)

        # (B,K)
        _, indices = torch.topk(
            scores,
            self.top_k,
            dim=1
        )

        # (B,K,C)
        gather_index = indices.unsqueeze(-1).expand(
            -1,
            -1,
            patch_tokens.size(-1)
        )

        selected_tokens = torch.gather(
            patch_tokens,
            dim=1,
            index=gather_index
        )

        return selected_tokens, scores, indices
