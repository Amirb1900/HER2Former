import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveDiversityTopKSelectorEx5(nn.Module):
    """
    Experiment 5
    --------------------------------------------------
    Adaptive Diversity-Aware Top-K Token Routing.

    Main idea:
        Token relevance is combined with an
        image-dependent diversity penalty.

    Unlike Experiment 4:
        diversity_weight is NOT fixed.

    Instead:

        lambda(x) =
            lambda_max -
            (lambda_max - lambda_min) * heterogeneity(x)

    Therefore:

        homogeneous image
            -> stronger redundancy suppression

        heterogeneous image
            -> weaker redundancy suppression

    Training:
        Differentiable soft Top-K mask.

    Inference:
        Hard Top-K indices.

    Input:
        patch_tokens: [B, N, C]

    Output:
        selected_tokens: [B, K, C]
        pooled_tokens:   [B, C]
        scores:          [B, N]
        indices:         [B, K]
        selection_weights: [B, N]
        heterogeneity:   [B]
        adaptive_lambda: [B]
    """

    def __init__(
        self,
        embed_dim=1280,
        top_k=64,
        lambda_min=0.05,
        lambda_max=0.50,
        temperature=0.5,
        threshold_iterations=20,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.top_k = top_k

        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        self.temperature = temperature
        self.threshold_iterations = threshold_iterations

        # ==================================================
        # Learnable HER2 relevance scorer
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
    # Image heterogeneity
    # ======================================================

    def _compute_heterogeneity(
        self,
        patch_tokens,
    ):
        """
        Estimate image-level morphological heterogeneity
        from the diversity of patch representations.

        High average similarity:
            -> low heterogeneity

        Low average similarity:
            -> high heterogeneity

        Returns:
            heterogeneity: [B]
        """

        normalized_tokens = F.normalize(
            patch_tokens,
            p=2,
            dim=-1
        )

        similarity = torch.bmm(
            normalized_tokens,
            normalized_tokens.transpose(1, 2)
        )

        N = patch_tokens.size(1)

        identity = torch.eye(
            N,
            device=patch_tokens.device,
            dtype=patch_tokens.dtype
        ).unsqueeze(0)

        # Remove diagonal.
        similarity = similarity * (1.0 - identity)

        # Only positive similarity contributes.
        positive_similarity = F.relu(
            similarity
        )

        # Mean pairwise similarity.
        mean_similarity = (
            positive_similarity.sum(
                dim=(1, 2)
            )
            /
            float(N * (N - 1))
        )

        # Convert similarity into heterogeneity.
        heterogeneity = 1.0 - mean_similarity

        # Numerical safety.
        heterogeneity = torch.clamp(
            heterogeneity,
            min=0.0,
            max=1.0
        )

        return heterogeneity

    # ======================================================
    # Adaptive diversity coefficient
    # ======================================================

    def _compute_adaptive_lambda(
        self,
        heterogeneity,
    ):
        """
        Convert image heterogeneity into an adaptive
        diversity coefficient.

        Low heterogeneity:
            -> lambda closer to lambda_max

        High heterogeneity:
            -> lambda closer to lambda_min
        """

        adaptive_lambda = (
            self.lambda_max
            -
            (
                self.lambda_max
                -
                self.lambda_min
            )
            *
            heterogeneity
        )

        return adaptive_lambda

    # ======================================================
    # Redundancy penalty
    # ======================================================

    def _compute_diversity_penalty(
        self,
        patch_tokens,
        relevance_weights,
    ):
        """
        Estimate redundancy for each token.

        A token receives a higher penalty when it is highly
        similar to other highly relevant tokens.

        Input:
            patch_tokens:
                [B, N, C]

            relevance_weights:
                [B, N]

        Output:
            penalty:
                [B, N]
        """

        normalized_tokens = F.normalize(
            patch_tokens,
            p=2,
            dim=-1
        )

        similarity = torch.bmm(
            normalized_tokens,
            normalized_tokens.transpose(1, 2)
        )

        N = patch_tokens.size(1)

        identity = torch.eye(
            N,
            device=patch_tokens.device,
            dtype=patch_tokens.dtype
        ).unsqueeze(0)

        similarity = similarity * (
            1.0 - identity
        )

        # Similarity below zero is not considered redundancy.
        similarity = F.relu(
            similarity
        )

        penalty = torch.bmm(
            similarity,
            relevance_weights.unsqueeze(-1)
        ).squeeze(-1)

        penalty = penalty / float(N - 1)

        return penalty

    # ======================================================
    # Differentiable K-Mass Top-K
    # ======================================================

    def _soft_topk_mask(
        self,
        scores,
    ):
        """
        Differentiable approximation of a K-element selection.

        The sigmoid threshold is found using bisection such
        that the total selection mass approaches K.
        """

        temperature = self.temperature

        low = (
            scores.min(
                dim=1,
                keepdim=True
            ).values
            - 10.0
        )

        high = (
            scores.max(
                dim=1,
                keepdim=True
            ).values
            + 10.0
        )

        for _ in range(
            self.threshold_iterations
        ):

            threshold = (
                low + high
            ) / 2.0

            mask = torch.sigmoid(
                (
                    scores - threshold
                )
                /
                temperature
            )

            count = mask.sum(
                dim=1,
                keepdim=True
            )

            # Too many selected.
            low = torch.where(
                count > self.top_k,
                threshold,
                low
            )

            # Too few selected.
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
            )
            /
            temperature
        )

        return mask

    # ======================================================
    # Forward
    # ======================================================

    def forward(
        self,
        patch_tokens,
    ):

        B, N, C = patch_tokens.shape

        # ==================================================
        # 1. HER2 relevance
        # ==================================================

        scores = self.score_layer(
            patch_tokens
        ).squeeze(-1)

        # ==================================================
        # 2. Initial relevance distribution
        # ==================================================

        relevance_weights = torch.softmax(
            scores,
            dim=1
        )

        # ==================================================
        # 3. Image heterogeneity
        # ==================================================

        heterogeneity = (
            self._compute_heterogeneity(
                patch_tokens
            )
        )

        # ==================================================
        # 4. Adaptive diversity coefficient
        # ==================================================

        adaptive_lambda = (
            self._compute_adaptive_lambda(
                heterogeneity
            )
        )

        # ==================================================
        # 5. Redundancy
        # ==================================================

        diversity_penalty = (
            self._compute_diversity_penalty(
                patch_tokens,
                relevance_weights
            )
        )

        # ==================================================
        # 6. Adaptive diversity-aware score
        # ==================================================

        adjusted_scores = (
            scores
            -
            adaptive_lambda.unsqueeze(1)
            *
            diversity_penalty
        )

        # ==================================================
        # 7. Differentiable Top-K
        # ==================================================

        selection_weights = (
            self._soft_topk_mask(
                adjusted_scores
            )
        )

        # ==================================================
        # 8. Soft selected representation
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

        pooled_tokens = torch.bmm(
            normalized_weights.unsqueeze(1),
            patch_tokens
        ).squeeze(1)

        # ==================================================
        # 9. Hard Top-K
        # ==================================================

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
        # 10. Return
        # ==================================================

        return (
            selected_tokens,
            pooled_tokens,
            adjusted_scores,
            indices,
            selection_weights,
            heterogeneity,
            adaptive_lambda,
        )
