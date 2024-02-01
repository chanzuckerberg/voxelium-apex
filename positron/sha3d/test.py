import time

import torch
from torch import nn

from positron.base.torch_utils import pca_dim_reduction


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

    for i in range(max_iter):
        # Compute the Gaussian kernel
        P = torch.exp(-D * beta.unsqueeze(1)) * (1 - torch.eye(n, device=device))

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

    # Normalize P matrix
    P = torch.exp(-D * beta.unsqueeze(1)) * (1 - torch.eye(n, device=device))
    P = P / torch.sum(P, dim=1) + eps
    P = P / P.sum()

    return P.detach()


class TSNE(nn.Module):
    def __init__(self, n_points, n_components, y=None):
        super(TSNE, self).__init__()
        self.n_points = n_points
        self.n_components = n_components
        if y is None:
            self.y = nn.Parameter(torch.randn(n_points, n_components) * 1e-4)
        else:
            self.y = nn.Parameter(y)

    def forward(self, P, eps=1e-12):
        sum_Y = torch.sum(self.y * self.y, dim=1)
        num = -2. * torch.mm(self.y, self.y.t())
        num = 1. / (1. + (sum_Y.unsqueeze(0) + num + sum_Y.unsqueeze(1)))
        num[range(self.n_points), range(self.n_points)] = 0.
        Q = num / (torch.sum(num) + eps)
        KL_divergence = torch.sum(P * torch.log((P + eps) / (Q + eps)))
        return KL_divergence


def do_tsne_test(x, n_components=2, perplexity=20.0):
    device = x.device
    t = time.time()
    P = compute_pairwise_affinities(x, perplexity)
    print(f"time {round(time.time() - t, 3)}")

    pca_x = pca_dim_reduction(x, n_components=n_components)
    tsne_model = TSNE(n_points=x.shape[0], n_components=n_components).to(device)
    optimizer = torch.optim.Adam(tsne_model.parameters(), lr=0.1)

    # Training loop
    for epoch in range(100):
        optimizer.zero_grad()
        tsne_loss = tsne_model(P)
        loss = tsne_loss #+ torch.nn.functional.mse_loss(tsne_model.y, pca_x) * 0.1
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, loss={loss.item()}')

    # Retrieve the optimized low-dimensional points
    return tsne_model.y
