"""
DanceOPD: On-Policy Generative Field Distillation — Synthetic 2D Data

Creates three "capability" distributions (teacher fields):
  - Teacher A: Two well-separated Gaussian blobs (class 0)
  - Teacher B: A ring / circular distribution (class 1)
  - Teacher C: Three diagonal Gaussian blobs (class 2)

Each teacher is trained to generate samples for its class via standard
flow matching.  The student is then distilled from all teachers via
the DanceOPD on-policy field distillation procedure.
"""

import torch
from torch.utils.data import Dataset


class GaussianMixtureDataset(Dataset):
    """Synthetic 2D Gaussian mixture for a single capability bucket."""

    def __init__(
        self,
        centers: list,
        std: float = 0.25,
        n_samples: int = 5000,
        seed: int = 42,
    ):
        super().__init__()
        self.centers = [torch.tensor(c, dtype=torch.float32) for c in centers]
        self.std = std
        self.n_samples = n_samples

        rng = torch.Generator().manual_seed(seed)
        dim = self.centers[0].shape[0]
        n_centers = len(self.centers)

        # Assign each sample to a random center, then sample around it
        assignments = torch.randint(0, n_centers, (n_samples,), generator=rng)
        offsets = torch.randn(n_samples, dim, generator=rng) * std

        data = torch.stack(
            [self.centers[assignments[i]] for i in range(n_samples)]
        ) + offsets
        self.data = data

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]


class RingDataset(Dataset):
    """Synthetic 2D ring distribution."""

    def __init__(self, radius: float = 2.0, std: float = 0.15, n_samples: int = 5000, seed: int = 42):
        super().__init__()
        self.n_samples = n_samples
        rng = torch.Generator().manual_seed(seed)
        angles = torch.rand(n_samples, generator=rng) * 2.0 * 3.14159265
        self.data = torch.stack([
            radius * torch.cos(angles) + std * torch.randn(n_samples, generator=rng),
            radius * torch.sin(angles) + std * torch.randn(n_samples, generator=rng),
        ], dim=-1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]


def build_capability_datasets(seed: int = 42):
    """Return a dict of {class_id: Dataset} for three capability buckets.

    Returns:
        dict mapping class label (int) → Dataset of 2D points.
    """
    datasets = {
        0: GaussianMixtureDataset(
            centers=[[-2.0, -2.0], [2.0, 2.0]],
            std=0.3,
            n_samples=5000,
            seed=seed,
        ),
        1: RingDataset(radius=2.0, std=0.15, n_samples=5000, seed=seed),
        2: GaussianMixtureDataset(
            centers=[[0.0, 3.0], [-3.0, -1.0], [3.0, -1.0]],
            std=0.3,
            n_samples=5000,
            seed=seed,
        ),
    }
    return datasets


def get_batch_from_bucket(datasets: dict, class_id: int, batch_size: int, device: torch.device):
    """Sample a random batch of data points from the specified bucket."""
    ds = datasets[class_id]
    indices = torch.randint(0, len(ds), (batch_size,))
    return ds.data[indices].to(device)
