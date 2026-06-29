"""
ShutterMuse Data Module — Synthetic Photography Guidance Data
===============================================================

Generates synthetic training data for both photographer-side and subject-side tasks.

Photographer-side data:
  - Random image crops with ground-truth aesthetic scores
  - 3-way decision labels (refine/keep/reject)
  - Ground-truth crop boxes for "refine" cases
  - Simulated salient object masks (for mask coverage reward)

Subject-side data:
  - Person-free scene images (simulated with colored backgrounds)
  - COCO-17 keypoint annotations
  - Visibility vectors (1=visible, 0=occluded, -1=out-of-frame)

Data format follows the paper's structured JSON output schema.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from PIL import Image
import random


# ─── Data Configuration ────────────────────────────────────────────────────────

@dataclass
class ShutterMuseDataConfig:
    """Data generation configuration."""
    image_size: int = 224
    num_samples: int = 1000
    vocab_size: int = 32000
    max_text_len: int = 64

    # Decision distribution: P(refine), P(keep), P(reject)
    decision_probs: Tuple[float, float, float] = (0.5, 0.3, 0.2)

    # COCO-17 keypoint names
    keypoint_names: Tuple[str, ...] = (
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    )


# ─── Sample Data Classes ──────────────────────────────────────────────────────

@dataclass
class PhotographerSample:
    """Single photographer-side training sample."""
    image: torch.Tensor          # (3, H, W) float tensor [0, 1]
    text_tokens: torch.Tensor    # (L,) int token IDs
    task_type: str               # "composition"
    decision: int                # 0=refine, 1=keep, 2=reject
    crop_box: Optional[torch.Tensor] = None   # (4,) [x1,y1,x2,y2]
    subject_mask: Optional[torch.Tensor] = None  # (H, W) binary mask
    aesthetic_score: float = 0.0


@dataclass
class SubjectSample:
    """Single subject-side training sample."""
    scene_image: torch.Tensor    # (3, H, W) person-free scene
    text_tokens: torch.Tensor    # (L,) int token IDs
    task_type: str               # "pose"
    keypoints: torch.Tensor      # (17, 2) normalized (x, y) in [0, 1]
    visibility: torch.Tensor     # (17,) values in {-1, 0, 1}


# ─── Synthetic Data Generation ────────────────────────────────────────────────

def generate_synthetic_image(
    size: int = 224,
    num_subjects: int = 1,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a synthetic image with a salient subject region.

    Returns:
        image: (3, H, W) RGB image tensor in [0, 1]
        subject_mask: (H, W) binary mask of the subject
        subject_box: (4,) [x1, y1, x2, y2] normalized subject location
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    img = torch.zeros(3, size, size)

    # Background: gradient
    bg_color = torch.rand(3)
    for c in range(3):
        img[c] = bg_color[c] + 0.1 * torch.linspace(0, 1, size).unsqueeze(1) * torch.ones(1, size)

    # Add random background texture
    noise = torch.randn(3, size, size) * 0.03
    img = img + noise

    # Subject region (elliptical)
    cx = torch.randint(size // 4, 3 * size // 4, (1,)).item()
    cy = torch.randint(size // 4, 3 * size // 4, (1,)).item()
    rx = torch.randint(size // 8, size // 3, (1,)).item()
    ry = torch.randint(size // 8, size // 3, (1,)).item()

    subject_color = torch.rand(3) * 0.5 + 0.3  # Distinct from background

    Y, X = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    ellipse = ((X - cx) ** 2 / (rx ** 2 + 1e-6) + (Y - cy) ** 2 / (ry ** 2 + 1e-6)) < 1

    subject_mask = ellipse.float()
    for c in range(3):
        img[c] = img[c] * (1 - subject_mask) + subject_color[c] * subject_mask

    # Normalized subject box
    x1 = max(0, (cx - rx) / size)
    y1 = max(0, (cy - ry) / size)
    x2 = min(1, (cx + rx) / size)
    y2 = min(1, (cy + ry) / size)
    subject_box = torch.tensor([x1, y1, x2, y2])

    img = img.clamp(0, 1)
    return img, subject_mask, subject_box


def generate_crop_box(
    subject_box: torch.Tensor,
    decision: int,
    min_margin: float = 0.05,
) -> Optional[torch.Tensor]:
    """
    Generate a crop box based on the decision.

    - refine: crop that tightens around the subject (with some variation)
    - keep: [0, 0, 1, 1] (full image)
    - reject: None (no valid crop)
    """
    if decision == 2:  # reject
        return None
    elif decision == 1:  # keep
        return torch.tensor([0.0, 0.0, 1.0, 1.0])
    else:  # refine
        sx1, sy1, sx2, sy2 = subject_box
        # Generate a crop that includes the subject but improves framing
        margin = random.uniform(0.05, 0.15)
        x1 = max(0, sx1 - margin * (sx2 - sx1))
        y1 = max(0, sy1 - margin * (sy2 - sy1))
        x2 = min(1, sx2 + margin * (sx2 - sx1))
        y2 = min(1, sy2 + margin * (sy2 - sy1))
        return torch.tensor([x1, y1, x2, y2])


def compute_aesthetic_score(
    crop_box: torch.Tensor,
    subject_box: torch.Tensor,
    decision: int,
) -> float:
    """
    Simulate aesthetic score based on crop quality.

    Higher score when crop is well-framed around subject.
    """
    if decision == 2:  # reject
        return random.uniform(0.0, 0.2)
    elif decision == 1:  # keep
        return random.uniform(0.7, 1.0)
    else:  # refine
        # IoU between crop and subject
        cx1, cy1, cx2, cy2 = crop_box
        sx1, sy1, sx2, sy2 = subject_box
        inter_x1 = max(cx1, sx1)
        inter_y1 = max(cy1, sy1)
        inter_x2 = min(cx2, sx2)
        inter_y2 = min(cy2, sy2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        crop_area = max(0.001, (cx2 - cx1) * (cy2 - cy1))
        subject_area = max(0.001, (sx2 - sx1) * (sy2 - sy1))
        union_area = crop_area + subject_area - inter_area
        iou = inter_area / union_area
        return min(1.0, 0.5 + iou * 0.5 + random.uniform(-0.1, 0.1))


def generate_coco_keypoints(
    image_size: int = 224,
    subject_box: torch.Tensor = None,
    occlusion_prob: float = 0.15,
    out_of_frame_prob: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic COCO-17 keypoints with realistic spatial structure.

    Returns:
        keypoints: (17, 2) normalized (x, y) in [0, 1]
        visibility: (17,) values in {-1, 0, 1}
    """
    if subject_box is None:
        subject_box = torch.tensor([0.2, 0.2, 0.8, 0.8])

    sx1, sy1, sx2, sy2 = subject_box
    cx = (sx1 + sx2) / 2
    cy = (sy1 + sy2) / 2
    bw = sx2 - sx1
    bh = sy2 - sy1

    # Define keypoint locations relative to subject center
    # (name, dx_frac, dy_frac) — relative offsets from center, as fraction of box size
    kp_defs = [
        ("nose",            0.0,   -0.35),
        ("left_eye",       -0.05,  -0.38),
        ("right_eye",       0.05,  -0.38),
        ("left_ear",       -0.12,  -0.35),
        ("right_ear",       0.12,  -0.35),
        ("left_shoulder",  -0.20,  -0.15),
        ("right_shoulder",  0.20,  -0.15),
        ("left_elbow",     -0.30,   0.05),
        ("right_elbow",     0.30,   0.05),
        ("left_wrist",     -0.35,   0.25),
        ("right_wrist",     0.35,   0.25),
        ("left_hip",       -0.12,   0.20),
        ("right_hip",       0.12,   0.20),
        ("left_knee",      -0.15,   0.45),
        ("right_knee",      0.15,   0.45),
        ("left_ankle",     -0.15,   0.70),
        ("right_ankle",     0.15,   0.70),
    ]

    keypoints = torch.zeros(17, 2)
    visibility = torch.ones(17, dtype=torch.long)

    for i, (name, dx, dy) in enumerate(kp_defs):
        x = cx + dx * bw + random.gauss(0, 0.02 * bw)
        y = cy + dy * bh + random.gauss(0, 0.02 * bh)

        # Random occlusion / out-of-frame
        r = random.random()
        if r < out_of_frame_prob:
            visibility[i] = -1  # out of frame
            x = random.choice([random.uniform(-0.1, -0.01), random.uniform(1.01, 1.1)])
            y = random.uniform(0, 1)
        elif r < out_of_frame_prob + occlusion_prob:
            visibility[i] = 0  # occluded
        else:
            visibility[i] = 1  # visible

        keypoints[i, 0] = x
        keypoints[i, 1] = y

    return keypoints, visibility


def generate_text_tokens(
    task_type: str,
    decision: Optional[int] = None,
    crop_box: Optional[torch.Tensor] = None,
    visibility: Optional[torch.Tensor] = None,
    vocab_size: int = 32000,
    max_len: int = 64,
) -> torch.Tensor:
    """
    Generate synthetic text token IDs for the response.

    In the real model, these are the target JSON output tokens.
    Here we generate plausible-length random token sequences.
    """
    # Use fixed seed tokens for different task types to make them distinguishable
    base_tokens = {
        "composition": [1, 42, 128, 256],    # composition task prefix
        "pose": [1, 42, 333, 512],           # pose task prefix
    }

    tokens = base_tokens.get(task_type, [1, 42])

    if decision == 0:  # refine
        tokens.extend([100, 200, 300])
    elif decision == 1:  # keep
        tokens.extend([101, 201])
    elif decision == 2:  # reject
        tokens.extend([102, 202, 302])

    # Fill remaining with random tokens (simulating text rationale)
    remaining = max_len - len(tokens)
    if remaining > 0:
        tokens.extend(random.choices(range(1000, vocab_size), k=remaining))

    # Truncate or pad
    tokens = tokens[:max_len]
    while len(tokens) < max_len:
        tokens.append(0)  # pad token

    return torch.tensor(tokens, dtype=torch.long)


# ─── Datasets ────────────────────────────────────────────────────────────────

class PhotographerSideDataset(Dataset):
    """
    Synthetic dataset for photographer-side photography guidance.

    Each sample contains:
      - A synthetic image with a subject
      - A 3-way decision (refine/keep/reject)
      - A crop box (for refine cases)
      - A subject mask (for mask coverage reward computation)
      - An aesthetic score
      - Text tokens (simulated JSON response)
    """

    def __init__(self, config: Optional[ShutterMuseDataConfig] = None):
        self.config = config or ShutterMuseDataConfig()
        self.samples: List[PhotographerSample] = []
        self._generate()

    def _generate(self):
        """Generate all synthetic samples."""
        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)

        for i in range(self.config.num_samples):
            img, mask, subject_box = generate_synthetic_image(
                self.config.image_size, seed=i
            )

            # Sample decision
            decision = random.choices(
                [0, 1, 2],
                weights=self.config.decision_probs,
                k=1
            )[0]

            crop_box = generate_crop_box(subject_box, decision)
            aesthetic_score = compute_aesthetic_score(
                crop_box if crop_box is not None else subject_box,
                subject_box,
                decision,
            )

            text_tokens = generate_text_tokens(
                "composition",
                decision=decision,
                crop_box=crop_box,
                vocab_size=self.config.vocab_size,
                max_len=self.config.max_text_len,
            )

            sample = PhotographerSample(
                image=img,
                text_tokens=text_tokens,
                task_type="composition",
                decision=decision,
                crop_box=crop_box,
                subject_mask=mask,
                aesthetic_score=aesthetic_score,
            )
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> PhotographerSample:
        return self.samples[idx]


class SubjectSideDataset(Dataset):
    """
    Synthetic dataset for subject-side pose guidance.

    Each sample contains:
      - A person-free scene image (simulated)
      - COCO-17 keypoints
      - Visibility vector
      - Text tokens (simulated pose rationale)
    """

    def __init__(self, config: Optional[ShutterMuseDataConfig] = None):
        self.config = config or ShutterMuseDataConfig()
        self.samples: List[SubjectSample] = []
        self._generate()

    def _generate(self):
        """Generate all synthetic samples."""
        random.seed(123)
        torch.manual_seed(123)
        np.random.seed(123)

        for i in range(self.config.num_samples):
            # Person-free scene: just a background image
            scene_img = torch.rand(3, self.config.image_size, self.config.image_size) * 0.5
            # Add some structure
            y_grad = torch.linspace(0, 0.3, self.config.image_size).unsqueeze(1)
            x_grad = torch.linspace(0, 0.1, self.config.image_size).unsqueeze(0)
            for c in range(3):
                scene_img[c] += y_grad + x_grad
            scene_img = scene_img.clamp(0, 1)

            # Subject box for keypoint placement
            subject_box = torch.tensor([
                random.uniform(0.2, 0.4),
                random.uniform(0.1, 0.3),
                random.uniform(0.6, 0.8),
                random.uniform(0.7, 0.95),
            ])

            keypoints, visibility = generate_coco_keypoints(
                self.config.image_size,
                subject_box=subject_box,
                occlusion_prob=0.15,
                out_of_frame_prob=0.1,
            )

            text_tokens = generate_text_tokens(
                "pose",
                visibility=visibility,
                vocab_size=self.config.vocab_size,
                max_len=self.config.max_text_len,
            )

            sample = SubjectSample(
                scene_image=scene_img,
                text_tokens=text_tokens,
                task_type="pose",
                keypoints=keypoints,
                visibility=visibility,
            )
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> SubjectSample:
        return self.samples[idx]


# ─── Collation Functions ──────────────────────────────────────────────────────

def photographer_collate_fn(batch: List[PhotographerSample]) -> Dict[str, torch.Tensor]:
    """Collate photographer-side samples into batches."""
    images = torch.stack([s.image for s in batch])
    text_tokens = torch.stack([s.text_tokens for s in batch])
    decisions = torch.tensor([s.decision for s in batch])

    crop_boxes = []
    masks = []
    for s in batch:
        if s.crop_box is not None:
            crop_boxes.append(s.crop_box)
        else:
            crop_boxes.append(torch.tensor([0.0, 0.0, 0.0, 0.0]))  # placeholder for reject
        masks.append(s.subject_mask)

    return {
        'image': images,
        'input_ids': text_tokens,
        'decision': decisions,
        'crop_box': torch.stack(crop_boxes),
        'subject_mask': torch.stack(masks),
        'task_type': 'composition',
    }


def subject_collate_fn(batch: List[SubjectSample]) -> Dict[str, torch.Tensor]:
    """Collate subject-side samples into batches."""
    images = torch.stack([s.scene_image for s in batch])
    text_tokens = torch.stack([s.text_tokens for s in batch])
    keypoints = torch.stack([s.keypoints for s in batch])
    visibility = torch.stack([s.visibility for s in batch])

    return {
        'image': images,
        'input_ids': text_tokens,
        'keypoints': keypoints,
        'visibility': visibility,
        'task_type': 'pose',
    }


# ─── Data Loading Utility ────────────────────────────────────────────────────

def create_dataloaders(
    config: Optional[ShutterMuseDataConfig] = None,
    batch_size: int = 4,
    num_workers: int = 0,
    train_split: float = 0.8,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Create train/val dataloaders for both tasks.

    Returns:
        photo_train_loader, photo_val_loader, subject_train_loader, subject_val_loader
    """
    cfg = config or ShutterMuseDataConfig()
    total = cfg.num_samples
    train_size = int(total * train_split)

    # Photographer-side
    photo_ds = PhotographerSideDataset(cfg)
    photo_train_ds, photo_val_ds = torch.utils.data.random_split(
        photo_ds, [train_size, total - train_size]
    )
    photo_train_loader = DataLoader(
        photo_train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=photographer_collate_fn, num_workers=num_workers,
    )
    photo_val_loader = DataLoader(
        photo_val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=photographer_collate_fn, num_workers=num_workers,
    )

    # Subject-side
    subject_ds = SubjectSideDataset(cfg)
    subject_train_ds, subject_val_ds = torch.utils.data.random_split(
        subject_ds, [train_size, total - train_size]
    )
    subject_train_loader = DataLoader(
        subject_train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=subject_collate_fn, num_workers=num_workers,
    )
    subject_val_loader = DataLoader(
        subject_val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=subject_collate_fn, num_workers=num_workers,
    )

    return photo_train_loader, photo_val_loader, subject_train_loader, subject_val_loader


if __name__ == "__main__":
    # Quick test of data generation
    print("Testing data generation...")
    config = ShutterMuseDataConfig(num_samples=100)

    # Test photographer-side
    photo_ds = PhotographerSideDataset(config)
    sample = photo_ds[0]
    print(f"\nPhotographer sample:")
    print(f"  Image shape: {sample.image.shape}")
    print(f"  Decision: {sample.decision} ({['refine', 'keep', 'reject'][sample.decision]})")
    print(f"  Crop box: {sample.crop_box}")
    print(f"  Aesthetic score: {sample.aesthetic_score:.3f}")
    print(f"  Mask shape: {sample.subject_mask.shape}")

    # Test subject-side
    subject_ds = SubjectSideDataset(config)
    sample = subject_ds[0]
    print(f"\nSubject sample:")
    print(f"  Image shape: {sample.scene_image.shape}")
    print(f"  Keypoints shape: {sample.keypoints.shape}")
    print(f"  Visibility: {sample.visibility.tolist()}")
    print(f"  Unique visibility values: {sorted(set(sample.visibility.tolist()))}")

    # Test dataloaders
    loaders = create_dataloaders(config, batch_size=8)
    photo_train, photo_val, subj_train, subj_val = loaders

    batch = next(iter(photo_train))
    print(f"\nPhotographer batch:")
    print(f"  Images: {batch['image'].shape}")
    print(f"  Decisions: {batch['decision'].shape}, values: {batch['decision'].tolist()}")
    print(f"  Crop boxes: {batch['crop_box'].shape}")

    batch = next(iter(subj_train))
    print(f"\nSubject batch:")
    print(f"  Images: {batch['image'].shape}")
    print(f"  Visibility: {batch['visibility'].shape}")
    print(f"  Keypoints: {batch['keypoints'].shape}")

    print("\n✅ Data generation test passed!")
