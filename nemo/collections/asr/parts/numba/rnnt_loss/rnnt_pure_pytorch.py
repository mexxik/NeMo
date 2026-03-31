"""
Pure PyTorch RNNT loss — no numba, no custom CUDA kernels.

Uses the log-sum-exp forward algorithm on GPU via standard PyTorch ops.
Slower than numba but works on all GPU architectures without crashes.
"""

import torch
import torch.nn.functional as F


def rnnt_loss_pure_pytorch(
    acts,           # [B, T, U, V+1] log-softmax'd
    labels,         # [B, U-1]
    input_lengths,  # [B]
    label_lengths,  # [B]
    blank,          # int
    reduction='mean',
):
    """
    Compute RNNT loss using pure PyTorch operations.

    Args:
        acts: Log-probabilities [B, T, U, V+1] (already log-softmax'd)
        labels: Target labels [B, max_U-1]
        input_lengths: Acoustic lengths [B]
        label_lengths: Label lengths [B]
        blank: Blank token index
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss tensor
    """
    B, maxT, maxU, _ = acts.shape
    device = acts.device
    dtype = acts.dtype

    costs = torch.zeros(B, device=device, dtype=dtype)

    for b in range(B):
        T = input_lengths[b].item()
        U = label_lengths[b].item() + 1  # +1 for blank prefix

        # Extract log-probs for this sample
        log_probs = acts[b, :T, :U, :]  # [T, U, V+1]
        lab = labels[b, :U - 1]          # [U-1]

        # Alpha forward pass: alpha[t, u] = log P(y_{1:u} | x_{1:t})
        alpha = torch.full((T, U), float('-inf'), device=device, dtype=dtype)
        alpha[0, 0] = 0.0

        # Fill first column: alpha[t, 0] = sum of blank emissions
        for t in range(1, T):
            alpha[t, 0] = alpha[t - 1, 0] + log_probs[t - 1, 0, blank]

        # Fill first row: alpha[0, u] = sum of label emissions
        for u in range(1, U):
            alpha[0, u] = alpha[0, u - 1] + log_probs[0, u - 1, lab[u - 1]]

        # Fill rest
        for t in range(1, T):
            for u in range(1, U):
                # Blank transition: alpha[t-1, u] + log P(blank | t-1, u)
                skip = alpha[t - 1, u] + log_probs[t - 1, u, blank]
                # Label transition: alpha[t, u-1] + log P(label[u-1] | t, u-1)
                emit = alpha[t, u - 1] + log_probs[t, u - 1, lab[u - 1]]
                alpha[t, u] = torch.logaddexp(skip, emit)

        # Final: emit blank at (T-1, U-1)
        log_likelihood = alpha[T - 1, U - 1] + log_probs[T - 1, U - 1, blank]
        costs[b] = -log_likelihood

    if reduction == 'mean':
        return costs.mean()
    elif reduction == 'sum':
        return costs.sum()
    return costs


class PurePytorchRNNTLossFunction(torch.autograd.Function):
    """Autograd function wrapping the pure PyTorch RNNT loss."""

    @staticmethod
    def forward(ctx, acts, labels, act_lens, label_lens, blank, reduction):
        # acts comes in as [B, T, U, V+1] raw logits
        # We need log-softmax along the vocabulary dimension
        log_probs = F.log_softmax(acts, dim=-1)

        # Compute loss with gradients
        loss = rnnt_loss_pure_pytorch(
            log_probs, labels, act_lens, label_lens,
            blank=blank, reduction=reduction,
        )
        return loss


class PurePytorchRNNTLoss(torch.nn.Module):
    """
    Drop-in replacement for NeMo's RNNTLossNumba / RNNTLossPytorch.

    Uses pure PyTorch ops — no numba, no custom CUDA kernels.
    Compatible with all GPU architectures.
    """

    def __init__(self, blank, reduction='mean_batch'):
        super().__init__()
        self.blank = blank
        # NeMo uses 'mean_batch' which is equivalent to 'mean'
        self.reduction = 'mean' if reduction == 'mean_batch' else (reduction or 'none')

    def forward(self, acts, labels, act_lens, label_lens):
        # acts: [B, T, U, V+1] — already log-softmax'd from NeMo's joint
        return rnnt_loss_pure_pytorch(
            acts, labels, act_lens, label_lens,
            blank=self.blank, reduction=self.reduction,
        )
