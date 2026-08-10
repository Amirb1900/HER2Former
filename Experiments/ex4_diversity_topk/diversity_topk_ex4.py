import torch
import torch.nn as nn
import torch.nn.functional as F


class DiversityAwareTopKEx4(nn.Module):
    """
    Diversity-Aware Top-K Token Selector
    Experiment 4

    Purpose
    -------
    Select K patch tokens that are:

        1. Highly relevant
        2. Diverse
        3. Less redundant

    Input
    -----
    patch_tokens:
        [B, N, D]

        B = batch size
        N = number of patch tokens (256)
        D = embedding dimension (1280)

    Output
    ------
    selected_tokens:
        [B, K, D]

    selected_scores:
        [B, K]

    selected_indices:
        [B, K]

    """

    def __init__(
        self,
        embed_dim=1280,
        top_k=64,
        diversity_weight=0.25,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.top_k = top_k
        self.diversity_weight = diversity_weight

        # --------------------------------------------------
        # Learnable relevance scorer
        # --------------------------------------------------

        self.score_layer = nn.Sequential(
            nn.LayerNorm(embed_dim),

            nn.Linear(
                embed_dim,
                embed_dim // 4
            ),

            nn.GELU(),

            nn.Linear(
                embed_dim // 4,
                1
            )
        )

    def forward(self, patch_tokens):
        """
        Parameters
        ----------
        patch_tokens:
            [B, N, D]

        Returns
        -------
        selected_tokens:
            [B, K, D]

        selected_scores:
            [B, K]

        selected_indices:
            [B, K]
        """

        B, N, D = patch_tokens.shape

        # ==================================================
        # 1. Relevance Score
        # ==================================================

        relevance_scores = self.score_layer(
            patch_tokens
        ).squeeze(-1)

        # [B, N]

        # Normalize scores

        relevance_scores = torch.sigmoid(
            relevance_scores
        )

        # ==================================================
        # 2. Initial candidate selection
        # ==================================================

        candidate_k = min(
            self.top_k * 2,
            N
        )

        candidate_scores, candidate_indices = torch.topk(
            relevance_scores,
            k=candidate_k,
            dim=1
        )

        # ==================================================
        # 3. Candidate tokens
        # ==================================================

        candidate_tokens = torch.gather(
            patch_tokens,
            1,
            candidate_indices.unsqueeze(-1).expand(
                -1,
                -1,
                D
            )
        )

        # ==================================================
        # 4. Normalize token representations
        # ==================================================

        normalized_tokens = F.normalize(
            candidate_tokens,
            p=2,
            dim=-1
        )

        # ==================================================
        # 5. Diversity-aware greedy selection
        # ==================================================

        selected_indices = []

        selected_token_indices = []

        for b in range(B):

            selected = []

            remaining = torch.arange(
                candidate_k,
                device=patch_tokens.device
            )

            # First token:
            # choose highest relevance

            first_idx = torch.argmax(
                candidate_scores[b]
            ).item()

            selected.append(
                first_idx
            )

            remaining = remaining[
                remaining != first_idx
            ]

            # --------------------------------------------------
            # Greedy diversity-aware selection
            # --------------------------------------------------

            while len(selected) < self.top_k:

                selected_tensor = torch.tensor(
                    selected,
                    device=patch_tokens.device,
                    dtype=torch.long
                )

                selected_features = normalized_tokens[
                    b,
                    selected_tensor
                ]

                remaining_features = normalized_tokens[
                    b,
                    remaining
                ]

                # Similarity to already selected tokens

                similarity = torch.matmul(
                    remaining_features,
                    selected_features.transpose(0, 1)
                )

                max_similarity = similarity.max(
                    dim=1
                ).values

                # --------------------------------------------------
                # Diversity-aware score
                # --------------------------------------------------

                relevance = candidate_scores[
                    b,
                    remaining
                ]

                final_score = (
                    relevance
                    -
                    self.diversity_weight *
                    max_similarity
                )

                best_position = torch.argmax(
                    final_score
                )

                best_candidate = remaining[
                    best_position
                ].item()

                selected.append(
                    best_candidate
                )

                remaining = remaining[
                    remaining != best_candidate
                ]

            selected_tensor = torch.tensor(
                selected,
                device=patch_tokens.device,
                dtype=torch.long
            )

            selected_token_indices.append(
                selected_tensor
            )

        # ==================================================
        # Stack selected candidate indices
        # ==================================================

        selected_token_indices = torch.stack(
            selected_token_indices,
            dim=0
        )

        # [B, K]

        # ==================================================
        # Convert candidate indices → original token indices
        # ==================================================

        selected_indices = torch.gather(
            candidate_indices,
            1,
            selected_token_indices
        )

        # ==================================================
        # Gather final tokens
        # ==================================================

        selected_tokens = torch.gather(
            patch_tokens,
            1,
            selected_indices.unsqueeze(-1).expand(
                -1,
                -1,
                D
            )
        )

        # ==================================================
        # Gather final relevance scores
        # ==================================================

        selected_scores = torch.gather(
            relevance_scores,
            1,
            selected_indices
        )

        return (
            selected_tokens,
            selected_scores,
            selected_indices
        )
