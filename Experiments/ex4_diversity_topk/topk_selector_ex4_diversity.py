import torch
import torch.nn as nn
import torch.nn.functional as F


class DiversityAwareTopKSelectorEx4(nn.Module):
    """
    Differentiable Diversity-Aware Top-K Token Routing.

    Experiment 4
    -------------
    Learns token relevance while discouraging redundant
    / highly similar patch tokens.

    Training
    --------
    Uses a differentiable soft selection mask whose
    total mass is approximately equal to top_k.

    Inference
    ---------
    Returns the actual hard Top-K token indices for
    interpretability and visualization.

    Input
    -----
    patch_tokens:
        [B, N, C]

    Output
    ------
    selected_tokens:
        [B, K, C]

    scores:
        [B, N]

    indices:
        [B, K]

    selection_weights:
        [B, N]
    """

    def __init__(
        self,
        embed_dim=1280,
        top_k=64,
        diversity_weight=0.25,
        temperature=0.5,
        threshold_iterations=20,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.top_k = top_k
        self.diversity_weight = diversity_weight
        self.temperature = temperature
        self.threshold_iterations = threshold_iterations

        # ==================================================
        # Learnable relevance scorer
        # ==================================================

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

    # ======================================================
    # Diversity penalty
    # ======================================================

    def _compute_diversity_penalty(
        self,
        patch_tokens,
        relevance_weights,
    ):
        """
        Compute redundancy penalty.

        Highly similar tokens receive a larger penalty
        when other similar tokens already have high
        relevance.

        patch_tokens:
            [B, N, C]

        relevance_weights:
            [B, N]

        Returns:
            diversity_penalty:
                [B, N]
        """

        normalized_tokens = F.normalize(
            patch_tokens,
            p=2,
            dim=-1
        )

        # Pairwise cosine similarity
        #
        # [B,N,C] x [B,C,N]
        #
        # -> [B,N,N]

        similarity = torch.bmm(
            normalized_tokens,
            normalized_tokens.transpose(1, 2)
        )

        # Remove self-similarity
        identity = torch.eye(
            similarity.size(-1),
            device=similarity.device,
            dtype=similarity.dtype
        ).unsqueeze(0)

        similarity = similarity * (
            1.0 - identity
        )

        # Only positive similarity contributes
        similarity = F.relu(
            similarity
        )

        # Weighted redundancy
        #
        # [B,N,N] @ [B,N,1]
        #
        # -> [B,N]

        penalty = torch.bmm(
            similarity,
            relevance_weights.unsqueeze(-1)
        ).squeeze(-1)

        # Normalize by number of tokens
        penalty = penalty / (
            patch_tokens.size(1) - 1
        )

        return penalty

    # ======================================================
    # Differentiable K-Mass Mask
    # ======================================================

    def _soft_topk_mask(
        self,
        scores,
    ):
        """
        Creates a differentiable selection mask.

        The threshold is found by differentiable bisection
        so that the sum of selection probabilities is
        approximately equal to K.

        scores:
            [B,N]

        Returns:
            mask:
                [B,N]
        """

        temperature = self.temperature

        low = scores.min(
            dim=1,
            keepdim=True
        ).values - 10.0

        high = scores.max(
            dim=1,
            keepdim=True
        ).values + 10.0

        for _ in range(
            self.threshold_iterations
        ):

            threshold = (
                low + high
            ) / 2.0

            mask = torch.sigmoid(
                (
                    scores - threshold
                ) / temperature
            )

            count = mask.sum(
                dim=1,
                keepdim=True
            )

            # Too many selected
            # -> increase threshold

            low = torch.where(
                count > self.top_k,
                threshold,
                low
            )

            # Too few selected
            # -> decrease threshold

            high = torch.where(
                count > self.top_k,
                high,
                threshold
            )

        threshold = (
            low + high
        ) / 2.0

        mask = torch.sigmoid(
            (
                scores - threshold
            ) / temperature
        )

        return mask

    # ======================================================
    # Forward
    # ======================================================

    def forward(
        self,
        patch_tokens,
    ):
        """
        Forward pass.

        patch_tokens:
            [B,N,C]

        Returns
        -------

        selected_tokens:
            [B,K,C]

        scores:
            [B,N]

        indices:
            [B,K]

        selection_weights:
            [B,N]
        """

        B, N, C = patch_tokens.shape

        # ==================================================
        # Relevance scores
        # ==================================================

        scores = self.score_layer(
            patch_tokens
        ).squeeze(-1)

        # ==================================================
        # Initial relevance probabilities
        # ==================================================

        relevance_weights = torch.softmax(
            scores,
            dim=1
        )

        # ==================================================
        # Diversity penalty
        # ==================================================

        diversity_penalty = (
            self._compute_diversity_penalty(
                patch_tokens,
                relevance_weights
            )
        )

        # ==================================================
        # Diversity-aware scores
        # ==================================================

        adjusted_scores = (
            scores
            -
            self.diversity_weight
            * diversity_penalty
        )

        # ==================================================
        # Differentiable Top-K mask
        # ==================================================

        selection_weights = self._soft_topk_mask(
            adjusted_scores
        )

        # ==================================================
        # Soft selected representation
        # ==================================================

        normalized_weights = (
            selection_weights
            /
            (
                selection_weights.sum(
                    dim=1,
                    keepdim=True
                )
                + 1e-8
            )
        )

        pooled = torch.bmm(
            normalized_weights.unsqueeze(1),
            patch_tokens
        ).squeeze(1)

        # ==================================================
        # Hard Top-K indices
        # ==================================================
        #
        # Used only for interpretation / visualization.
        #

        _, indices = torch.topk(
            adjusted_scores,
            self.top_k,
            dim=1
        )

        gather_index = (
            indices
            .unsqueeze(-1)
            .expand(-1, -1, C)
        )

        selected_tokens = torch.gather(
            patch_tokens,
            dim=1,
            index=gather_index
        )

        # ==================================================
        # Return
        # ==================================================

        return (
            selected_tokens,
            pooled,
            adjusted_scores,
            indices,
            selection_weights
        )
