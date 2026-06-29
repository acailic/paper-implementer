"""
Synthetic 2D Physics Scene Dataset.

Generates simple physics scenes (bouncing balls, falling objects) with:
- RGB video frames
- Ground-truth point trajectories (analytical)
- Depth maps (from z-coordinates)

These replace CoTracker3, Depth-Anything-V2, and V-JEPA 2 outputs
in the full PhysisForcing pipeline.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random


class PhysicsScene:
    """
    A single 2D physics scene with circular objects.
    Simulates Newtonian mechanics: gravity, elastic collisions, boundary bouncing.
    """

    def __init__(self, W=64, H=64, n_frames=16, dt=0.05, gravity=9.8):
        self.W = W
        self.H = H
        self.n_frames = n_frames
        self.dt = dt
        self.gravity = gravity
        self.objects = []  # list of dicts: {x, y, vx, vy, radius, color, z}
        self._seed = None

    def reset(self, seed=None):
        """Initialize a random scene."""
        if seed is not None:
            self._seed = seed
        rng = np.random.RandomState(self._seed)

        self.objects = []
        n_objects = rng.randint(2, 6)

        for _ in range(n_objects):
            radius = rng.uniform(3, 8)
            x = rng.uniform(radius + 2, self.W - radius - 2)
            y = rng.uniform(radius + 2, self.H - radius - 2)
            vx = rng.uniform(-15, 15)
            vy = rng.uniform(-15, 15)
            z = rng.uniform(0.1, 1.0)  # depth (0=near, 1=far)

            # Random bright color
            hue = rng.uniform(0, 1)
            color = self._hsv_to_rgb(hue, 0.8, 0.9)

            self.objects.append({
                'x': x, 'y': y, 'vx': vx, 'vy': vy,
                'radius': radius, 'color': color, 'z': z,
            })

        return self

    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB (numpy array of shape (3,))."""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return np.array([r, g, b])

    def step(self):
        """Advance one timestep with physics."""
        dt = self.dt

        for obj in self.objects:
            # Gravity
            obj['vy'] += self.gravity * dt

            # Update position
            obj['x'] += obj['vx'] * dt
            obj['y'] += obj['vy'] * dt

            # Boundary collisions (elastic)
            r = obj['radius']
            restitution = 0.85

            if obj['x'] - r < 0:
                obj['x'] = r
                obj['vx'] = abs(obj['vx']) * restitution
            elif obj['x'] + r > self.W:
                obj['x'] = self.W - r
                obj['vx'] = -abs(obj['vx']) * restitution

            if obj['y'] - r < 0:
                obj['y'] = r
                obj['vy'] = abs(obj['vy']) * restitution
            elif obj['y'] + r > self.H:
                obj['y'] = self.H - r
                obj['vy'] = -abs(obj['vy']) * restitution

        # Object-object elastic collisions
        n = len(self.objects)
        for i in range(n):
            for j in range(i + 1, n):
                self._collide(self.objects[i], self.objects[j])

    def _collide(self, a, b):
        """Handle elastic collision between two circular objects."""
        dx = b['x'] - a['x']
        dy = b['y'] - a['y']
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = a['radius'] + b['radius']

        if dist < min_dist and dist > 0:
            # Normalize collision vector
            nx = dx / dist
            ny = dy / dist

            # Relative velocity along collision normal
            dvx = a['vx'] - b['vx']
            dvy = a['vy'] - b['vy']
            dvn = dvx * nx + dvy * ny

            if dvn > 0:  # Objects approaching
                restitution = 0.9
                a['vx'] -= restitution * dvn * nx
                a['vy'] -= restitution * dvn * ny
                b['vx'] += restitution * dvn * nx
                b['vy'] += restitution * dvn * ny

                # Separate objects
                overlap = min_dist - dist
                a['x'] -= overlap * 0.5 * nx
                a['y'] -= overlap * 0.5 * ny
                b['x'] += overlap * 0.5 * nx
                b['y'] += overlap * 0.5 * ny

    def get_trajectory(self, grid_size=8):
        """
        Get ground-truth point trajectories on a regular grid.
        Returns trajectories of points that are on/near objects.
        
        Returns:
            trajectories: (N_pts, T, 2) — (x, y) positions
        """
        # Create a grid of query points
        xs = np.linspace(0, self.W - 1, grid_size)
        ys = np.linspace(0, self.H - 1, grid_size)
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_points = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)

        # Run simulation to get all frames
        positions_per_frame = []
        positions_per_frame.append(self._get_point_positions(grid_points))

        for _ in range(self.n_frames - 1):
            self.step()
            positions_per_frame.append(self._get_point_positions(grid_points))

        trajectories = np.stack(positions_per_frame, axis=1)  # (N_pts, T, 2)
        return trajectories

    def _get_point_positions(self, grid_points):
        """
        For each grid point, find the nearest object and track it.
        Returns positions for all grid points.
        """
        positions = np.copy(grid_points)  # Default: stay in place

        for i, (gx, gy) in enumerate(grid_points):
            # Find if this point is inside any object
            best_dist = float('inf')
            best_obj = None
            for obj in self.objects:
                dist = np.sqrt((gx - obj['x']) ** 2 + (gy - obj['y']) ** 2)
                if dist < obj['radius'] * 1.5 and dist < best_dist:
                    best_dist = dist
                    best_obj = obj

            # If point is near an object, track the object's center
            # (offset by original displacement)
            if best_obj is not None:
                # Compute original offset from object center
                # For simplicity, just track object center
                positions[i, 0] = best_obj['x']
                positions[i, 1] = best_obj['y']

        return positions

    def render_frame(self):
        """
        Render current state to RGB image.
        Returns: (H, W, 3) float array in [-1, 1]
        """
        img = np.full((self.H, self.W, 3), -0.5, dtype=np.float32)  # dark gray bg

        # Sort by depth (far objects first)
        sorted_objs = sorted(self.objects, key=lambda o: -o['z'])

        for obj in sorted_objs:
            r = int(obj['radius'])
            cx, cy = int(round(obj['x'])), int(round(obj['y']))
            color = obj['color'] * 2 - 1  # scale to [-1, 1]

            # Draw filled circle
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    px, py = cx + dx, cy + dy
                    if 0 <= px < self.W and 0 <= py < self.H:
                        if dx * dx + dy * dy <= r * r:
                            # Add slight shading for 3D effect
                            shade = 1.0 - 0.3 * (dx * dx + dy * dy) / (r * r + 1e-6)
                            img[py, px] = color * shade

        return img

    def get_depth_map(self):
        """
        Get depth map from z-coordinates.
        Returns: (H, W) float — 0=near, 1=far
        """
        depth = np.ones((self.H, self.W), dtype=np.float32)

        for obj in self.objects:
            r = int(obj['radius'])
            cx, cy = int(round(obj['x'])), int(round(obj['y']))

            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    px, py = cx + dx, cy + dy
                    if 0 <= px < self.W and 0 <= py < self.H:
                        if dx * dx + dy * dy <= r * r:
                            depth[py, px] = obj['z']

        return depth


class PhysicsSceneDataset(Dataset):
    """
    Dataset of synthetic 2D physics scenes.
    Each sample returns:
    - video: (T, C, H, W) normalized to [-1, 1]
    - trajectories: (N_pts, T, 2) ground-truth point tracks
    - depth_map: (H, W) depth values
    """

    def __init__(
        self,
        n_samples=1000,
        H=64,
        W=64,
        n_frames=16,
        trajectory_grid=8,
        seed=42,
    ):
        self.n_samples = n_samples
        self.H = H
        self.W = W
        self.n_frames = n_frames
        self.trajectory_grid = trajectory_grid
        self.rng = random.Random(seed)

        # Pre-generate all scenes
        self.scenes = []
        for i in range(n_samples):
            scene_seed = seed + i * 7 + 13
            self.scenes.append(scene_seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        seed = self.scenes[idx]
        rng = np.random.RandomState(seed)

        # Create scene
        scene = PhysicsScene(
            W=self.W, H=self.H,
            n_frames=self.n_frames, dt=0.05
        )
        scene.reset(seed=seed)

        # Collect all frames
        frames = []
        for t in range(self.n_frames):
            if t > 0:
                scene.step()
            frame = scene.render_frame()
            frames.append(frame)

        video = np.stack(frames, axis=0)  # (T, H, W, 3)
        video = np.transpose(video, (0, 3, 1, 2))  # (T, C, H, W)

        # Get depth map from first frame
        depth_map = scene.get_depth_map()  # (H, W)

        # Get trajectories (re-run simulation for clean tracking)
        scene2 = PhysicsScene(
            W=self.W, H=self.H,
            n_frames=self.n_frames, dt=0.05
        )
        scene2.reset(seed=seed)
        trajectories = scene2.get_trajectory(grid_size=self.trajectory_grid)

        return {
            'video': torch.from_numpy(video).float(),
            'trajectories': torch.from_numpy(trajectories).float(),
            'depth_map': torch.from_numpy(depth_map).float(),
        }


def get_dataloaders(
    n_train=800,
    n_val=200,
    H=64,
    W=64,
    n_frames=16,
    trajectory_grid=8,
    batch_size=8,
    num_workers=0,
    seed=42,
):
    """Create train and validation dataloaders."""
    train_dataset = PhysicsSceneDataset(
        n_samples=n_train,
        H=H, W=W,
        n_frames=n_frames,
        trajectory_grid=trajectory_grid,
        seed=seed,
    )
    val_dataset = PhysicsSceneDataset(
        n_samples=n_val,
        H=H, W=W,
        n_frames=n_frames,
        trajectory_grid=trajectory_grid,
        seed=seed + 10000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
