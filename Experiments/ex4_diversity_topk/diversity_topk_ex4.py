import torch
import torch.nn as nn
import torch.nn.functional as F


class DiversityAwareTopKEx4(nn.Module):
    """
    Diversity-Aware Top-K Token Selector
    Experiment 4

    This module is a controlled modification of the
    standard TopKSelector used in Experiment 2.

    Experiment 2:
        score = Linear(token)
        select Top-K based on relevance

    Experiment 4:
        score = Linear(token)
        select Top-K using:

            relevance - diversity penalty

    The relevance scorer is intentionally identical
    to the baseline TopKSelector.

    Input
    -----
    patch_tokens:
        [B, N, C]

    Output
    ------
    selected_tokens:
        [B, K, C]

    selected_scores:
        [B, N]

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

        self.top_k = top_k

        self.diversity_weight = diversity_weight

        # ==================================================
        # IMPORTANT:
        # Same scorer as EXP2
        # ==================================================

        self.score_layer = nn.Linear(
            embed_dim,
            1
        )

    def forward(
        self,
        patch_tokens
    ):
        """
        Parameters
        ----------
        patch_tokens:
            [B, N, C]

        Returns
        -------
        selected_tokens:
            [B, K, C]

        scores:
            [B, N]

        selected_indices:
            [B, K]
        """

        B, N, C = patch_tokens.shape

        # ==================================================
        # 1. Relevance Scores
        # ==================================================

        scores = self.score_layer(
            patch_tokens
        ).squeeze(-1)

        # [B, N]

        # ==================================================
        # 2. Normalize token representations
        # ==================================================

        normalized_tokens = F.normalize(
            patch_tokens,
            p=2,
            dim=-1
        )

        # [B, N, C]

        # ==================================================
        # 3. Diversity-Aware Greedy Selection
        # ==================================================

        batch_selected_indices = []

        for b in range(B):

            # --------------------------------------------------
            # First token:
            # highest relevance
            # --------------------------------------------------

            first_index = torch.argmax(
                scores[b]
            )

            selected = [
                first_index
            ]

            # --------------------------------------------------
            # Iteratively select remaining K-1 tokens
            # --------------------------------------------------

            for _ in range(
                1,
                self.top_k
            ):

                selected_tensor = torch.stack(
                    selected
                )

                # [S, C]
                selected_features = normalized_tokens[
                    b,
                    selected_tensor
                ]

                # --------------------------------------------------
                # Similarity between all tokens and selected tokens
                # --------------------------------------------------

                similarity = torch.matmul(
                    normalized_tokens[b],
                    selected_features.transpose(
                        0,
                        1
                    )
                )

                # [N, S]

                max_similarity = similarity.max(
                    dim=1
                ).values

                # --------------------------------------------------
                # Diversity-aware score
                # --------------------------------------------------

                diversity_score = (
                    scores[b]
                    -
                    self.diversity_weight *
                    max_similarity
                )

                # --------------------------------------------------
                # Prevent already selected tokens
                # from being selected again
                # --------------------------------------------------

                diversity_score[
                    selected_tensor
                ] = -float("inf")

                # --------------------------------------------------
                # Select next token
                # --------------------------------------------------

                next_index = torch.argmax(
                    diversity_score
                )

                selected.append(
                    next_index
                )

            # --------------------------------------------------
            # Stack selected indices
            # --------------------------------------------------

            selected = torch.stack(
                selected
            )

            batch_selected_indices.append(
                selected
            )

        # ==================================================
        # 4. Final indices
        # ==================================================

        selected_indices = torch.stack(
            batch_selected_indices
        )

        # [B, K]

        # ==================================================
        # 5. Gather selected tokens
        # ==================================================

        gather_index = (
            selected_indices
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                C
            )
        )

        selected_tokens = torch.gather(
            patch_tokens,
            dim=1,
            index=gather_index
        )

        # [B, K, C]

        return (
            selected_tokens,
            scores,
            selected_indices
        )
