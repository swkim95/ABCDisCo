"""Distance correlation utilities used to decorrelate classifier outputs.

This module implements the loss term introduced in the DisCo paper
(`https://arxiv.org/abs/1905.08628`).  The original implementation contains
very short variable names that can be difficult to interpret when reusing the
code in a different project.  The functions below therefore include extensive
docstrings and inline comments that explain the different steps of the
calculation while retaining the behaviour of the reference implementation.
"""

import numpy as np
import torch


def distance_corr(variable_1, variable_2, normedweight, power=1):
    """Return the (weighted) distance correlation between two 1-D tensors.

    Args:
        variable_1 (torch.Tensor): First variable to decorrelate (e.g. jet mass).
        variable_2 (torch.Tensor): Second variable to decorrelate (e.g. network
            score).  Both inputs must be one-dimensional tensors with an equal
            number of elements.
        normedweight (torch.Tensor): Per-example weight.  The weights are
            expected to sum to the number of samples, which makes the loss
            comparable to the unweighted case.
        power (int): Exponent used when constructing the distance correlation.
            ``power=1`` corresponds to the usual definition, while ``power=2``
            avoids the square root.

    Returns:
        torch.Tensor: Scalar tensor containing the distance correlation.  The
        value is differentiable and can be added directly to a loss function,
        e.g. ``total_loss = bce_loss + lambda * distance_corr(...)``.
    """

    # Compute the pairwise absolute differences |x_i - x_j| for both input
    # variables.  The resulting matrices have shape (N, N) where N is the
    # number of examples.
    xx = variable_1.view(-1, 1).repeat(1, len(variable_1)).view(
        len(variable_1), len(variable_1)
    )
    yy = variable_1.repeat(len(variable_1), 1).view(len(variable_1), len(variable_1))
    amat = (xx - yy).abs()

    xx = variable_2.view(-1, 1).repeat(1, len(variable_2)).view(
        len(variable_2), len(variable_2)
    )
    yy = variable_2.repeat(len(variable_2), 1).view(len(variable_2), len(variable_2))
    bmat = (xx - yy).abs()

    # Subtract the (weighted) row and column means from each distance matrix to
    # construct the centred matrices A and B, as in Eq. (2.2) of the DisCo
    # paper.  The weights are applied per element to allow uneven datasets.
    amatavg = torch.mean(amat * normedweight, dim=1)
    Amat = (
        amat
        - amatavg.repeat(len(variable_1), 1)
        .view(len(variable_1), len(variable_1))
        - amatavg.view(-1, 1)
        .repeat(1, len(variable_1))
        .view(len(variable_1), len(variable_1))
        + torch.mean(amatavg * normedweight)
    )

    bmatavg = torch.mean(bmat * normedweight, dim=1)
    Bmat = (
        bmat
        - bmatavg.repeat(len(variable_2), 1)
        .view(len(variable_2), len(variable_2))
        - bmatavg.view(-1, 1)
        .repeat(1, len(variable_2))
        .view(len(variable_2), len(variable_2))
        + torch.mean(bmatavg * normedweight)
    )

    # Compute the weighted averages that form the numerator and denominators of
    # the distance correlation definition.
    ABavg = torch.mean(Amat * Bmat * normedweight, dim=1)
    AAavg = torch.mean(Amat * Amat * normedweight, dim=1)
    BBavg = torch.mean(Bmat * Bmat * normedweight, dim=1)

    if power == 1:
        dCorr = (torch.mean(ABavg * normedweight)) / torch.sqrt(
            torch.mean(AAavg * normedweight) * torch.mean(BBavg * normedweight)
        )
    elif power == 2:
        dCorr = (torch.mean(ABavg * normedweight)) ** 2 / (
            torch.mean(AAavg * normedweight) * torch.mean(BBavg * normedweight)
        )
    else:
        dCorr = (
            torch.mean(ABavg * normedweight)
            / torch.sqrt(
                torch.mean(AAavg * normedweight)
                * torch.mean(BBavg * normedweight)
            )
        ) ** power
    return dCorr


def distance_corr_unbiased(variable_1, variable_2, normedweight, power=1):
    """Unbiased distance correlation estimator with weighted inputs.

    The original distance correlation has a small positive bias for finite
    samples.  This function implements the unbiased estimator from
    Eq. (3.5) of the DisCo paper while retaining support for per-example
    weights.  The arguments are identical to :func:`distance_corr`.
    """

    xx = variable_1.view(-1, 1).repeat(1, len(variable_1)).view(
        len(variable_1), len(variable_1)
    )
    yy = variable_1.repeat(len(variable_1), 1).view(len(variable_1), len(variable_1))
    amat = (xx - yy).abs()

    xx = variable_2.view(-1, 1).repeat(1, len(variable_2)).view(
        len(variable_2), len(variable_2)
    )
    yy = variable_2.repeat(len(variable_2), 1).view(len(variable_2), len(variable_2))
    bmat = (xx - yy).abs()

    # The unbiased estimator replaces the means in the centring operation by
    # sums with factors depending on the sample size (N-1 and N-2).  The weight
    # tensor is used in the same way as in the biased estimator.
    amatavg = 1 / (len(variable_1) - 2) * torch.sum(amat * normedweight, dim=1)
    Amat = (
        amat
        - amatavg.repeat(len(variable_1), 1)
        .view(len(variable_1), len(variable_1))
        - amatavg.view(-1, 1)
        .repeat(1, len(variable_1))
        .view(len(variable_1), len(variable_1))
        + 1 / (len(variable_1) - 1) * torch.sum(amatavg * normedweight)
    )

    bmatavg = 1 / (len(variable_1) - 2) * torch.sum(bmat * normedweight, dim=1)
    Bmat = (
        bmat
        - bmatavg.repeat(len(variable_2), 1)
        .view(len(variable_2), len(variable_2))
        - bmatavg.view(-1, 1)
        .repeat(1, len(variable_2))
        .view(len(variable_2), len(variable_2))
        + 1 / (len(variable_1) - 1) * torch.sum(bmatavg * normedweight)
    )

    # The unbiased form requires the diagonals of the centred matrices to be
    # set to zero explicitly; otherwise the correction terms are invalidated.
    Amat.fill_diagonal_(0)
    Bmat.fill_diagonal_(0)

    ABavg = torch.mean(Amat * Bmat * normedweight, dim=1)
    AAavg = torch.mean(Amat * Amat * normedweight, dim=1)
    BBavg = torch.mean(Bmat * Bmat * normedweight, dim=1)

    if power == 1:
        denom = torch.mean(AAavg * normedweight) * torch.mean(BBavg * normedweight)
        dCorr = torch.tensor(0.0, device=variable_1.device)
        if denom > 0:
            dCorr = (torch.mean(ABavg * normedweight)) / torch.sqrt(denom)
    elif power == 2:
        dCorr = (torch.mean(ABavg * normedweight)) ** 2 / (
            torch.mean(AAavg * normedweight) * torch.mean(BBavg * normedweight)
        )
    else:
        dCorr = (
            torch.mean(ABavg * normedweight)
            / torch.sqrt(
                torch.mean(AAavg * normedweight)
                * torch.mean(BBavg * normedweight)
            )
        ) ** power
    return dCorr


# ---------------------------------------------------------------------------
# Memory efficient (but slow) reference implementations
# ---------------------------------------------------------------------------

def dcovsq_unbiased_slow(variable_1, variable_2, normedweight):
    """Slow but memory-friendly unbiased distance covariance estimator.

    This routine follows Eqs. (3.1) and (3.2) of the unbiased estimator in
    ``arXiv:1310.2926``.  Unlike :func:`distance_corr`, it avoids constructing
    ``N x N`` matrices and therefore uses significantly less memory for large
    sample sizes.  The trade-off is speed because the implementation uses
    explicit Python loops.
    """

    term1 = 0.0
    term2 = 0.0
    term3a = 0.0
    term3b = 0.0
    for i in range(len(variable_1)):
        # Using NumPy here is slightly more accurate than torch for the
        # element-wise absolute differences when operating in double precision.
        amat_vec = np.abs(variable_1 - variable_1[i])
        bmat_vec = np.abs(variable_2 - variable_2[i])
        term1 += np.mean(amat_vec * bmat_vec)
        term2 += np.mean(amat_vec) * np.mean(bmat_vec)
        term3a += np.mean(amat_vec)
        term3b += np.mean(bmat_vec)

    dCovsq = term1 - 2 / (len(variable_1) - 2) * len(variable_1) * term2
    dCovsq += (
        len(variable_1)
        * term3a
        * term3b
        / ((len(variable_1) - 2) * (len(variable_1) - 1))
    )

    return dCovsq


def dcorrsq_unbiased_slow(variable_1, variable_2, normedweight):
    """Slow unbiased distance correlation (squared) using the covariance helper."""

    numerator = dcovsq_unbiased_slow(variable_1, variable_2, normedweight)
    denominator = np.sqrt(
        dcovsq_unbiased_slow(variable_1, variable_1, normedweight)
        * dcovsq_unbiased_slow(variable_2, variable_2, normedweight)
    )

    dCorrsq = numerator / denominator

    return dCorrsq
