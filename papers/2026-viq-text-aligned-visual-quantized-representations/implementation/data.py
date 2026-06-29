"""
ViQ Data Module — CIFAR-10 dataset with text-aligned labels.

Since CIFAR-10 has class names (not free-text answers), we construct synthetic
text supervision: given an image, the "text query" is "What is in this image?"
and the "answer" is the class name token. This mimics the VQA-style text
alignment loss used in Stage 1 of ViQ.

For the reconstruction branch (Stage 2-1), we use a small VAE trained
end-to-end as a proxy for the Qwen-Image VAE in the original paper.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Simple tokenizer: map each class name to a token ID
CLASS_TO_ID = {name: i for i, name in enumerate(CIFAR10_CLASSES)}
ID_TO_CLASS = {i: name for name, i in CLASS_TO_ID.items()}


class TextAlignedCIFAR10(Dataset):
    """CIFAR-10 wrapped with text-aligned supervision for ViQ training.

    Each sample returns:
        image:       [3, 32, 32] Tensor
        class_id:    int (0-9)
        text_tokens: [num_tokens] LongTensor — token IDs for the answer
        query_tokens: [num_tokens] LongTensor — token IDs for the query
    """

    QUERY_TEMPLATE = "What is in this image?"

    def __init__(self, root="./data", train=True, image_size=32):
        self.dataset = datasets.CIFAR10(
            root=root, train=train, download=True,
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2470, 0.2435, 0.2616)),
            ]),
        )
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        class_id = label
        # Text tokens: we use a simple learned embedding for class IDs
        text_tokens = torch.tensor([class_id], dtype=torch.long)
        # Query is fixed; encoded as a special token
        query_tokens = torch.tensor([10], dtype=torch.long)  # 10 = "query" token
        return image, class_id, text_tokens, query_tokens


def get_dataloaders(batch_size=128, image_size=32, num_workers=2):
    """Create train and test dataloaders for CIFAR-10."""
    train_dataset = TextAlignedCIFAR10(train=True, image_size=image_size)
    test_dataset = TextAlignedCIFAR10(train=False, image_size=image_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader, len(CIFAR10_CLASSES)


# ---------------------------------------------------------------------------
# Tiny VAE — proxy for the Qwen-Image VAE used in the paper for reconstruction
# ---------------------------------------------------------------------------

class TinyVAE(torch.nn.Module):
    """Minimal convolutional VAE for generating latent targets (recon branch).

    Architecture:
        Encoder: 3×(Conv→BN→ReLU) → Flatten → FC → (μ, log_σ)
        Decoder: FC → Unflatten → 3×(ConvT→BN→ReLU) → ConvT → Sigmoid
    """

    def __init__(self, latent_dim=128, image_size=32, in_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),  # 32→16
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, stride=2, padding=1),          # 16→8
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 4, stride=2, padding=1),         # 8→4
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.flat_dim = 128 * 4 * 4  # 2048
        self.fc_mu = torch.nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = torch.nn.Linear(self.flat_dim, latent_dim)

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, self.flat_dim),
            torch.nn.ReLU(),
            torch.nn.Unflatten(1, (128, 4, 4)),
            torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 4→8
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 8→16
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),  # 16→32
            torch.nn.Sigmoid(),
        )

    def encode(self, x):
        """Return (μ, log_σ) from input image."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = μ + σ·ε."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent back to image [0, 1]."""
        return self.decoder(z)

    def forward(self, x):
        """Full forward: x → z → x̂. Returns (x̂, μ, log_σ)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def get_latent(self, x):
        """Get deterministic latent (μ) for reconstruction target."""
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu

    @staticmethod
    def vae_loss(x_recon, x, mu, logvar, kl_weight=1e-4):
        """VAE loss = Reconstruction MSE + KL divergence."""
        recon_loss = torch.nn.functional.mse_loss(x_recon, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + kl_weight * kl_loss) / x.size(0)


def pretrain_vae(device="cuda", epochs=20, batch_size=256, latent_dim=128):
    """Pre-train the tiny VAE on CIFAR-10 training set.

    Returns a frozen VAE ready to serve as the reconstruction target encoder.
    """
    train_dataset = TextAlignedCIFAR10(train=True, image_size=32)
    # Use unnormalized images for VAE (VAE decoder outputs [0,1])
    train_dataset.dataset.transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    vae = TinyVAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    vae.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (image, *_) in enumerate(loader):
            image = image.to(device)
            x_recon, mu, logvar = vae(image)
            loss = TinyVAE.vae_loss(x_recon, image, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / (batch_idx + 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[VAE Pretrain] Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

    # Freeze — will only be used as a target encoder
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print("[VAE Pretrain] Done. VAE is frozen.")
    return vae
