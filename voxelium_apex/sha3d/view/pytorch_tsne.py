"""
Implementation of the tSNE algorithm using Pytorch autograd.
This is not an optimal implementation.
Rather it is very easy to follow the math.
However, due to the use of vectorized calls it is still very fast.
"""

import time

import torch
from torch import nn


@torch.no_grad()
def compute_pairwise_affinities(x, perplexity, max_iter=50, eps=1e-12, tol=1e-2):
    """
    Compute pairwise affinities using Gaussian distribution around each point, optimized with PyTorch tensor operations.
    """

    device = x.device
    n = x.size(0)

    # Compute squared Euclidean distance matrix efficiently
    sum_X = torch.sum(x ** 2, dim=1)
    D = sum_X.unsqueeze(0) - 2 * torch.mm(x, x.t()) + sum_X.unsqueeze(1)

    # Prevent numerical instability
    D = torch.clamp(D, min=0)

    # Log of perplexity
    log_perplexity = torch.log(torch.tensor(perplexity))

    beta = torch.ones(n, device=device)

    # Initialize beta_min and beta_max tensors
    beta_min = torch.full_like(beta, float('-inf'))
    beta_max = torch.full_like(beta, float('inf'))

    diagonal_idx = torch.arange(0, x.size(0), device=x.device).unsqueeze(1)

    for i in range(max_iter):
        # Compute the Gaussian kernel
        P = torch.exp(-D * beta.unsqueeze(1)) * (1 - torch.eye(n, device=device))

        # Set diagonal to zeros
        P.scatter_(1, diagonal_idx, 0)

        # Sum of elements per row
        sumP = torch.sum(P, dim=1) + eps
        P = P / sumP.unsqueeze(1)

        # Compute entropy
        entropy = torch.log(sumP) + beta * torch.sum(torch.multiply(D, P), dim=1) / sumP

        # Desired entropy is log(perplexity)
        entropy_diff = entropy - log_perplexity

        # Check if all entropy_diff values are within tolerance
        if torch.all(torch.abs(entropy_diff) <= tol) or i == max_iter - 1:
            break

        # Update beta values - vectorized binary search
        # Adjust beta based on the condition
        increase = entropy_diff > 0
        decrease = ~increase

        beta_min[increase] = beta[increase]
        beta_max[decrease] = beta[decrease]

        # Update beta - if betamax is inf, double beta; else take the average
        beta[increase] = torch.where(torch.isinf(beta_max[increase]),
                                     beta[increase] * 2,
                                     (beta[increase] + beta_max[increase]) / 2)

        # Update beta - if betamin is inf, halve beta; else take the average
        beta[decrease] = torch.where(torch.isinf(beta_min[decrease]),
                                     beta[decrease] / 2,
                                     (beta[decrease] + beta_min[decrease]) / 2)

    # Compute the Gaussian kernel
    P = torch.exp(-D * beta.unsqueeze(1))

    # Set diagonal to zeros
    P.scatter_(1, diagonal_idx, 0)

    # Sum of elements per row
    P = P / (torch.sum(P, dim=1, keepdim=True) + eps)

    # Normalize P matrix
    P = P / P.sum()

    return P.detach()


class TSNE(nn.Module):
    def __init__(self, n_points, n_components, y_init=None):
        super(TSNE, self).__init__()
        self.n_points = n_points
        self.n_components = n_components

        if y_init is None:
            self.y = nn.Parameter(torch.randn(n_points, n_components) * 1e-4)
        else:
            if y_init.size(0) != n_points or y_init.size(1) != n_components:
                RuntimeError("Bad starting Y dimensions")
            self.y = nn.Parameter(y_init.clone())

    def forward(self, P, eps=1e-12):
        # Calculate the t-Student distribution
        sum_Y = torch.sum(self.y * self.y, dim=1)
        num = -2. * torch.mm(self.y, self.y.t())
        num = 1. / (1. + (sum_Y.unsqueeze(0) + num + sum_Y.unsqueeze(1)))

        # Set diagonal to zeros
        idx = torch.arange(0, self.n_points, device=num.device).unsqueeze(1)
        num.scatter_(1, idx, 0)

        # Convert to probability
        Q = num / (torch.sum(num) + eps)

        # Calculate the KL-divergence
        KL_divergence = torch.sum(P * torch.log((P + eps) / (Q + eps)))

        return KL_divergence

    def fit(self, x, perplexity=200.0, lr=1, verbose=False):
        t = time.time()
        P = compute_pairwise_affinities(x, perplexity)
        if verbose:
            print(f"time {round(time.time() - t, 3)}")

        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        # Training loop
        t = time.time()

        last_loss = float('inf')
        patience = 0

        for epoch in range(300):
            optimizer.zero_grad()
            tsne_loss = self(P)
            loss = tsne_loss * x.size(0)
            loss.backward()
            optimizer.step()

            if loss > last_loss * 0.99:
                patience += 1
            else:
                patience = 0

            if patience > 20:
                if verbose:
                    print(f'Converged at epoch {epoch}, loss={tsne_loss.item()}')
                break

            if verbose and epoch % 10 == 0:
                print(f'Epoch {epoch}, loss={tsne_loss.item()}')

            last_loss = loss.item()

        if verbose:
            print(f"time {round(time.time() - t, 3)}")

        # Retrieve the optimized low-dimensional points
        return self.y


def apply_tsne(x, y_init=None, n_components=2, perplexity=200.0, lr=1, verbose=False):
    tsne = TSNE(n_points=x.size(0), n_components=n_components, y_init=y_init)
    tsne = tsne.to(x.device)
    return tsne.fit(
        x=x, perplexity=perplexity, lr=lr, verbose=verbose
    )
