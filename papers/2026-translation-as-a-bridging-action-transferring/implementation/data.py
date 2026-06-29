"""
Synthetic data generation for 'Translation as a Bridging Action'.

Generates paired human-robot demonstrations for a simple 2D manipulation task:
  - Objects on a 2D table, camera observes from above
  - Actions: move end-effector to object, grasp, move to target, release
  - Human data: noisy wrist poses (with deliberately noisy rotations)
  - Robot data: clean end-effector poses + gripper signals
  - Bridging action: relative wrist translation in camera frame

The synthetic data simulates the key properties of real paired data:
  1. Humans and robots perform the same task with different embodiments
  2. Human wrist rotation is noisy (realistic from pose estimators)
  3. Bridging action (translation-only) is shared and reliable
  4. Different data sources have different available action components
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import math


# ─── SE(3) helpers ────────────────────────────────────────────────────────────

def make_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4×4 SE(3) matrix from rotation R (3×3) and translation t (3,)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rotation_z(angle: float) -> np.ndarray:
    """Rotation matrix around Z-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def random_rotation(noise_std: float = 0.1) -> np.ndarray:
    """Random rotation with small perturbation (axis-angle approx)."""
    axis = np.random.randn(3)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    angle = np.random.randn() * noise_std
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float64)
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    return R


# ─── Bridging Action Extraction ────────────────────────────────────────────────

def extract_bridging_action(
    wrist_poses_t: np.ndarray,
    wrist_poses_future: np.ndarray,
    cam_pose_t: np.ndarray,
) -> np.ndarray:
    """
    Extract bridging action a^{3D-wrist}: relative wrist translation in camera frame.

    W^c_{t+i} = (T^{c←w}_t)^{-1} · W^w_{t+i}
    a^{3D-wrist}_t = T(W^c_{t+i}) - T(W^c_t)  for i = 1, ..., k

    Args:
        wrist_poses_t: (4, 4) wrist pose at time t in world frame
        wrist_poses_future: (k, 4, 4) wrist poses at future timesteps in world frame
        cam_pose_t: (4, 4) camera extrinsic at time t (world→camera)

    Returns:
        bridging: (k, 3) relative translation in camera frame
    """
    T_inv = np.linalg.inv(cam_pose_t)
    W_c_t = T_inv @ wrist_poses_t
    trans_t = W_c_t[:3, 3]

    k = wrist_poses_future.shape[0]
    bridging = np.zeros((k, 3), dtype=np.float32)
    for i in range(k):
        W_c_ti = T_inv @ wrist_poses_future[i]
        bridging[i] = W_c_ti[:3, 3] - trans_t

    return bridging


def extract_eef_action(
    wrist_poses_t: np.ndarray,
    wrist_poses_future: np.ndarray,
) -> np.ndarray:
    """
    Extract 6DoF end-effector action a^{6D-eef}.

    a^{6D-eef}_t = (W^w_t)^{-1} · W^w_{t+i}

    Parameterized as (dx, dy, dz, droll, dpitch, dyaw) via SE(3) logarithm.

    Args:
        wrist_poses_t: (4, 4) end-effector pose at time t
        wrist_poses_future: (k, 4, 4) future poses

    Returns:
        eef: (k, 6) relative pose parameterization
    """
    inv_Wt = np.linalg.inv(wrist_poses_t)
    k = wrist_poses_future.shape[0]
    eef = np.zeros((k, 6), dtype=np.float32)
    for i in range(k):
        rel = inv_Wt @ wrist_poses_future[i]
        eef[i, :3] = rel[:3, 3]  # translation
        # Extract Euler angles from rotation matrix (ZYX convention)
        R = rel[:3, :3]
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        if sy > 1e-6:
            eef[i, 3] = np.arctan2(R[2, 1], R[2, 2])
            eef[i, 4] = np.arctan2(-R[2, 0], sy)
            eef[i, 5] = np.arctan2(R[1, 0], R[0, 0])
        else:
            eef[i, 3] = np.arctan2(-R[1, 2], R[1, 1])
            eef[i, 4] = np.arctan2(-R[2, 0], sy)
            eef[i, 5] = 0.0
    return eef


# ─── 2D Manipulation Environment ─────────────────────────────────────────────

class TableTop2D:
    """
    Simple 2D tabletop manipulation environment for synthetic data generation.

    Objects at known positions, pick-and-place trajectories generated.
    Camera is fixed overhead.
    """

    def __init__(self, table_size=1.0, cam_height=0.5, workspace_range=0.4):
        self.table_size = table_size
        self.cam_height = cam_height
        self.workspace = workspace_range
        # Camera pose: looking down from above, at origin
        self.cam_pose = make_se3(
            rotation_z(np.pi),           # rotated 180° (camera facing down)
            np.array([0, 0, cam_height]) # positioned above table
        )

    def generate_trajectory(
        self, start_pos, target_pos, num_steps=20, noise_std=0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a pick-and-place trajectory from start to target.

        Returns:
            positions: (num_steps, 3) XYZ positions along trajectory
            grippers:  (num_steps,) binary gripper states (0=open, 1=close)
        """
        positions = np.zeros((num_steps, 3), dtype=np.float64)
        grippers = np.zeros(num_steps, dtype=np.float32)

        # Phase 1: Move to object (open gripper)
        phase1 = num_steps // 3
        for i in range(phase1):
            t = i / max(phase1 - 1, 1)
            # Smooth interpolation (ease-in-out)
            t_smooth = 0.5 * (1 - np.cos(np.pi * t))
            positions[i] = start_pos + t_smooth * (np.array([0, 0, 0]) - start_pos)
            grippers[i] = 0.0

        # Phase 2: Grasp and lift
        phase2 = num_steps // 3
        for i in range(phase2):
            idx = phase1 + i
            t = i / max(phase2 - 1, 1)
            t_smooth = 0.5 * (1 - np.cos(np.pi * t))
            mid = np.array([0.0, 0.0, 0.05])  # lift slightly
            positions[idx] = t_smooth * mid
            grippers[idx] = min(1.0, t * 2)  # close gripper

        # Phase 3: Move to target and release
        phase3 = num_steps - phase1 - phase2
        for i in range(phase3):
            idx = phase1 + phase2 + i
            t = i / max(phase3 - 1, 1)
            t_smooth = 0.5 * (1 - np.cos(np.pi * t))
            mid = np.array([0.0, 0.0, 0.05])
            positions[idx] = mid + t_smooth * (target_pos - mid)
            grippers[idx] = max(0.0, 1.0 - t * 2)  # open gripper

        # Add noise if requested
        if noise_std > 0:
            positions += np.random.randn(*positions.shape) * noise_std

        return positions, grippers

    def generate_sample(self) -> Dict[str, np.ndarray]:
        """
        Generate a single demonstration sample.

        Returns:
            dict with 'image', 'wrist_poses', 'eef_poses', 'grippers',
                 'bridging', 'eef_action', 'language'
        """
        # Random pick and place
        start_xy = (np.random.rand(2) - 0.5) * 2 * self.workspace
        target_xy = (np.random.rand(2) - 0.5) * 2 * self.workspace

        # Generate trajectory
        num_steps = 20
        start_pos = np.array([start_xy[0], start_xy[1], 0.15])
        target_pos = np.array([target_xy[0], target_xy[1], 0.15])

        positions, grippers = self.generate_trajectory(
            start_pos, target_pos, num_steps, noise_std=0.002
        )

        # Build SE(3) poses (no rotation for simplicity in 2D task)
        base_rotation = rotation_z(np.random.uniform(0, 2 * np.pi))
        wrist_poses = np.zeros((num_steps, 4, 4), dtype=np.float64)
        for i in range(num_steps):
            wrist_poses[i] = make_se3(base_rotation, positions[i])

        # Generate synthetic observation image (simple 2D rendering)
        image = self._render_observation(positions, start_xy, target_xy)

        # Language instruction
        language = self._generate_language(start_xy, target_xy)

        return {
            'image': image,                          # (3, 64, 64)
            'wrist_poses': wrist_poses,              # (T, 4, 4)
            'grippers': grippers,                    # (T,)
            'cam_pose': self.cam_pose,               # (4, 4)
            'language': language,                     # string
        }

    def _render_observation(
        self, positions: np.ndarray, start_xy: np.ndarray, target_xy: np.ndarray
    ) -> np.ndarray:
        """Simple synthetic image: colored dots for object, target, and EE."""
        img = np.ones((3, 64, 64), dtype=np.float32) * 0.9  # light background

        def draw_circle(img, cx, cy, r, color):
            """Draw a filled circle on the image."""
            y_range = np.arange(64).reshape(-1, 1)
            x_range = np.arange(64).reshape(1, -1)
            mask = (y_range - cy) ** 2 + (x_range - cx) ** 2 <= r ** 2
            for c in range(3):
                img[c] = np.where(mask, color[c], img[c])

        # Convert world coords to pixel coords
        # Table is [-0.4, 0.4] → pixels [8, 56]
        def world_to_pixel(xy):
            px = int((xy[0] / self.workspace + 1) * 28 + 4)
            py = int((xy[1] / self.workspace + 1) * 28 + 4)
            return px, py

        # Draw start object (red)
        sx, sy = world_to_pixel(start_xy)
        draw_circle(img, sx, sy, 4, [0.8, 0.2, 0.2])

        # Draw target location (green)
        tx, ty = world_to_pixel(target_xy)
        draw_circle(img, tx, ty, 4, [0.2, 0.8, 0.2])

        # Draw current EE position (blue)
        current = positions[0]
        ex, ey = world_to_pixel(current[:2])
        draw_circle(img, ex, ey, 3, [0.2, 0.2, 0.8])

        return img

    def _generate_language(self, start_xy: np.ndarray, target_xy: np.ndarray) -> str:
        """Generate simple language instruction."""
        instructions = [
            "pick up the object and place it at the target",
            "move the red object to the green location",
            "grasp the item and put it down over there",
        ]
        return instructions[np.random.randint(len(instructions))]


# ─── Human Data Simulator ─────────────────────────────────────────────────────

class HumanDataSimulator:
    """
    Simulates noisy human demonstration data.

    Key simulation properties:
    1. Wrist translation is reliable (ground truth + small noise)
    2. Wrist rotation is very noisy (simulating hand pose estimator errors)
    3. In-the-wild humans have no gripper signal
    4. In-lab humans have manually annotated gripper signals
    """

    def __init__(self, env: TableTop2D, rotation_noise_std: float = 0.4):
        self.env = env
        self.rotation_noise_std = rotation_noise_std

    def generate_human_sample(
        self, in_lab: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate a human demonstration sample.

        Args:
            in_lab: if True, include gripper signal (in-lab annotation)

        Returns:
            dict with all data fields
        """
        sample = self.env.generate_sample()
        T = sample['wrist_poses'].shape[0]

        # Add noisy rotation to wrist poses (simulating HPE noise)
        human_poses = sample['wrist_poses'].copy()
        for i in range(T):
            noisy_rot = sample['wrist_poses'][i, :3, :3] @ random_rotation(
                self.rotation_noise_std
            )
            human_poses[i, :3, :3] = noisy_rot
            # Add small translation noise
            human_poses[i, :3, 3] += np.random.randn(3) * 0.005

        sample['wrist_poses'] = human_poses
        sample['data_source'] = 'lab_human' if in_lab else 'wild_human'

        return sample

    def generate_robot_sample(self) -> Dict[str, np.ndarray]:
        """Generate a clean robot demonstration sample."""
        sample = self.env.generate_sample()
        sample['data_source'] = 'robot'
        return sample


# ─── Dataset ─────────────────────────────────────────────────────────────────

class BridgingActionDataset(Dataset):
    """
    PyTorch dataset for bridging action training.

    For each sample, extracts:
      - Bridging action (always): relative wrist translation in camera frame
      - EEF action (robot only): relative 6DoF end-effector action
      - Gripper action (robot + lab_human): binary gripper signals

    Returns tokenized language and action chunks for training.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        action_chunk_size: int = 4,
        human_ratio: float = 0.6,
        lab_human_ratio: float = 0.2,
        rotation_noise_std: float = 0.4,
    ):
        super().__init__()
        self.action_chunk_size = action_chunk_size

        env = TableTop2D()
        human_sim = HumanDataSimulator(env, rotation_noise_std)

        # Generate mixed dataset
        n_human = int(num_samples * human_ratio)
        n_lab = int(num_samples * lab_human_ratio)
        n_robot = num_samples - n_human - n_lab

        self.samples = []
        print(f"  Generating {n_human} wild human, {n_lab} lab human, "
              f"{n_robot} robot samples...")

        for _ in range(n_human):
            self.samples.append(human_sim.generate_human_sample(in_lab=False))
        for _ in range(n_lab):
            self.samples.append(human_sim.generate_human_sample(in_lab=True))
        for _ in range(n_robot):
            self.samples.append(human_sim.generate_robot_sample())

        # Simple tokenization: map characters to integers
        self.char_to_idx = self._build_vocab()
        self.max_lang_len = 32

    def _build_vocab(self) -> Dict[str, int]:
        """Build simple character-level vocabulary."""
        vocab = {}
        idx = 1  # 0 reserved for padding
        for sample in self.samples:
            for ch in sample['language']:
                if ch not in vocab:
                    vocab[ch] = idx
                    idx += 1
        return vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        data_source = sample['data_source']
        k = self.action_chunk_size

        # Extract observation image at time t (use first frame)
        image = torch.from_numpy(sample['image']).float()  # (3, 64, 64)

        # Tokenize language
        lang_tokens = self._tokenize(sample['language'])  # (max_lang_len,)

        # Choose a random starting timestep for action chunk
        T = sample['wrist_poses'].shape[0]
        t_start = np.random.randint(0, T - k)
        wrist_t = sample['wrist_poses'][t_start]
        wrist_future = sample['wrist_poses'][t_start + 1: t_start + 1 + k]
        cam_pose = sample['cam_pose']

        # Extract bridging action (always present)
        bridging = extract_bridging_action(wrist_t, wrist_future, cam_pose)
        bridging_flat = bridging.reshape(-1).astype(np.float32)  # (k*3,)

        # Extract 6DoF EEF action (robot only)
        eef_flat = np.zeros(k * 6, dtype=np.float32)
        if data_source == 'robot':
            eef = extract_eef_action(wrist_t, wrist_future)
            eef_flat = eef.reshape(-1).astype(np.float32)

        # Extract gripper action (robot + lab_human)
        gripper_flat = np.zeros(k * 1, dtype=np.float32)
        if data_source in ('robot', 'lab_human'):
            grippers = sample['grippers'][t_start + 1: t_start + 1 + k]
            gripper_flat = grippers.astype(np.float32)

        return {
            'image': image,
            'lang_tokens': lang_tokens,
            'bridging': torch.from_numpy(bridging_flat),
            'eef': torch.from_numpy(eef_flat),
            'gripper': torch.from_numpy(gripper_flat),
            'data_source': data_source,
        }

    def _tokenize(self, text: str) -> torch.Tensor:
        """Simple character-level tokenization."""
        tokens = [self.char_to_idx.get(ch, 1) for ch in text]
        tokens = tokens[:self.max_lang_len]
        # Pad to max_lang_len
        tokens += [0] * (self.max_lang_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)


def create_dataloaders(
    num_train: int = 2000,
    num_val: int = 200,
    batch_size: int = 32,
    action_chunk_size: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    train_dataset = BridgingActionDataset(
        num_samples=num_train,
        action_chunk_size=action_chunk_size,
    )
    val_dataset = BridgingActionDataset(
        num_samples=num_val,
        action_chunk_size=action_chunk_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=0)
    return train_loader, val_loader


# ─── Random Bridging Substitution (Stage II training strategy) ───────────────

def apply_bridging_substitution(
    batch: Dict[str, torch.Tensor],
    substitution_prob: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    On robot samples, randomly substitute a^{3D-wrist} for a^{6D-eef} as
    the prediction target. This is the key training strategy from the paper
    that forces the model to ground bridging representations into executable
    robot actions.

    When substituting, the EEF target is replaced with the bridging target
    (padded to EEF dimensions), so the flow matching loss trains the EEF head
    to predict the bridging action instead.

    Without this substitution (ablation): success drops 38.33% → 12.50%.

    Args:
        batch: dict with 'bridging', 'eef', 'gripper', 'data_source'
        substitution_prob: probability of substitution per robot sample

    Returns:
        Modified batch (in-place for efficiency)
    """
    B = batch['bridging'].shape[0]
    for i in range(B):
        if batch['data_source'][i] != 'robot':
            continue
        if np.random.rand() < substitution_prob:
            # Substitute: use bridging as EEF target
            # Pad bridging (k*3) to EEF dimensions (k*6)
            b = batch['bridging'][i]
            k = b.shape[0] // 3
            b_reshaped = b.reshape(k, 3)
            # Pad with zeros for rotation dimensions
            eef_sub = torch.cat([
                b_reshaped,
                torch.zeros(k, 3, device=b.device)
            ], dim=-1).reshape(-1)
            batch['eef'][i] = eef_sub
            batch['substituted'] = True  # flag for loss computation
    return batch
