import torch
import torch.nn.functional as F


@torch.no_grad()
def get_knn(points, k, return_indices=True, return_distances=False, batch_size=None):
    device = points.device
    n = points.size(0)

    if return_indices:
        indices = torch.zeros(n, k, dtype=torch.long, device=device)

    if return_distances:
        distances = torch.zeros(n, k, dtype=torch.float32, device=device)

    if batch_size is None:
        # Calculate an optimal batch size
        max_product = 1e7
        batch_size = min(max_product // n, n)
        batch_size = int(max(batch_size, 1))

    for i in range(0, n, batch_size):
        # Correctly handle the last batch which might be smaller than batch_size
        batch_end = min(i + batch_size, n)
        batch = points[i:batch_end]

        # Compute distances
        dist = torch.cdist(batch, points)

        # Vectorized way to set distances to self to infinity
        batch_indices = torch.arange(i, batch_end, device=device).unsqueeze(1)
        dist.scatter_(1, batch_indices, float('inf'))

        # Find the k-nearest neighbors
        batch_distances, batch_indices = torch.topk(dist, k, largest=False)

        if return_indices:
            indices[i:batch_end] = batch_indices

        if return_distances:
            distances[i:batch_end] = batch_distances

    if return_indices and return_distances:
        return indices, distances

    if return_indices:
        return indices

    if return_distances:
        return distances


