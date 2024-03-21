#!/usr/bin/env python3

import torch
import torch.nn.functional as F


def bsc_loss(anchor, target, temperature=2.0):
    batch_size = anchor.shape[0]

    anchor = F.normalize(anchor, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)

    # Compute cosine similarity matrix
    similarity_matrix = torch.matmul(anchor, target.t())

    # Get positive and negative logits
    mask = torch.eye(batch_size, dtype=torch.bool).to(anchor.device)
    positive_logits = similarity_matrix[mask]
    negative_logits = similarity_matrix[~mask].reshape(batch_size, -1)

    # Compute numerator and denominator for softmax
    numerator = torch.exp(positive_logits / temperature)
    denominator = torch.sum(torch.exp(negative_logits / temperature), dim=1, keepdim=True)

    # Compute loss
    loss = torch.log(numerator / denominator)
    loss = -torch.mean(loss)

    return loss


def n_plus_one_tuplet_loss(anchor, target, weight=None):
    batch_size = anchor.shape[0]

    # scale = 2 / (anchor.std() + target.std() + 1e-3)
    # anchor *= scale.detach()
    # target *= scale.detach()

    anchor = F.normalize(anchor, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)

    # Compute cosine similarity matrix
    similarity_matrix = torch.matmul(anchor, target.t())

    # Get positive and negative logits
    mask = torch.eye(batch_size, dtype=torch.bool).to(anchor.device)
    positive = similarity_matrix[mask]
    negative = similarity_matrix[~mask].reshape(batch_size, -1)

    # Compute the loss for each example with softmax
    losses = torch.log(1 + torch.sum(torch.exp(negative - positive[:, None]), dim=1))

    if weight is not None:
        losses *= weight

    return losses.mean()


def n_plus_one_tuplet_loss2(anchor, target):
    batch_size = anchor.shape[0]

    anchor = F.normalize(anchor, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)

    # Compute cosine similarity matrix
    similarity_matrix = torch.matmul(anchor, target.t())

    # Get positive and negative logits
    mask = torch.eye(batch_size, dtype=torch.bool).to(anchor.device)
    positive = similarity_matrix[mask]
    negative = similarity_matrix[~mask].reshape(batch_size, -1)

    # Compute the loss for each example
    losses = torch.mean(torch.relu(negative - positive[:, None]), dim=1)

    return losses.mean()


def cosine_similarity_loss(anchor, target):
    # Normalize
    anchor = F.normalize(anchor, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)

    # Calculate loss
    return 1 - torch.mean(anchor * target)


def cosine_similarity_loss2(anchor, target):
    # Normalize
    anchor = F.normalize(anchor, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)

    # Calculate loss
    return torch.log(1 + torch.sum(torch.exp(-anchor * target), dim=1)).mean()


def similarity_loss(anchor, target):
    norm = torch.cat([anchor, target], 0).std(0, keepdim=True) + 1e-12
    anchor = anchor / norm
    target = target / norm
    return (anchor - target).square().mean()


def batch_triplet_loss(anchor, target, margin=0.1):
    batch_size = anchor.shape[0]

    # Calculate distances
    distances = torch.cdist(anchor, target, p=2)

    # Get positive and negative logits
    mask = torch.eye(batch_size, dtype=torch.bool).to(anchor.device)
    positive = distances[mask]
    negative = distances[~mask].reshape(batch_size, -1)

    # Calculate loss
    loss = torch.relu(positive[:, None] - negative + margin)
    loss = loss.square().mean()

    return loss


def tsne_loss(A, B, bandwidth=0.1, eps=1e-12):
    """
    Compute the KL-divergence loss for clustering, based on a Gaussian distribution
    of distances.
    """
    # Compute pairwise distances using torch.cdist
    A = A / (A.std() + 1e-12)
    B = B / (B.std() + 1e-12)

    dist_A = torch.cdist(A, A, p=2) / (bandwidth + 1e-12)
    dist_B = torch.cdist(B, B, p=2) / (bandwidth + 1e-12)

    # Convert distances to probabilities
    P = torch.softmax(-dist_A.square(), dim=1)
    Q = torch.softmax(-dist_B.square(), dim=1)

    # Use the KL-divergence formula
    # Adding a small epsilon to avoid log(0)
    kl_divergence = P * torch.log(P / (Q + eps) + eps)

    # Sum over pairs of points, and average over the batch
    loss = kl_divergence.sum(dim=(1,)).mean()

    return loss


def triplet_loss(anchor, target, margin=1.0, p=2, eps=1e-6):
    negative = torch.roll(target, 1, 0)
    return torch.nn.functional.triplet_margin_loss(anchor, target, negative, margin=margin, p=p, eps=eps)
