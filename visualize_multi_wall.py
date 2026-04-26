"""Quick script to visually inspect the multi-wall environment."""

import matplotlib.pyplot as plt
import torch

from eb_jepa.datasets.two_rooms.env import DotWall
from eb_jepa.datasets.two_rooms.wall_dataset import WallDatasetConfig

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for i, n_walls in enumerate([1, 2, 3]):
    for j in range(2):
        config = WallDatasetConfig(
            n_walls=n_walls,
            min_wall_spacing=12,
            fix_wall=False,
            device=torch.device("cpu"),
        )
        env = DotWall(config, normalize=False, n_allowed_steps=400)
        obs, info = env.reset()

        # obs is (2, H, W): channel 0 = dot, channel 1 = walls
        dot_ch = obs[0].cpu().numpy()
        wall_ch = obs[1].cpu().numpy()

        # Combine into RGB: walls in white, dot in green, background black
        h, w = dot_ch.shape
        rgb = torch.zeros(h, w, 3)
        rgb[:, :, 0] = torch.tensor(wall_ch / 255.0)  # walls white
        rgb[:, :, 1] = torch.tensor(wall_ch / 255.0)
        rgb[:, :, 2] = torch.tensor(wall_ch / 255.0)
        rgb[:, :, 1] = torch.clamp(rgb[:, :, 1] + torch.tensor(dot_ch / 255.0), 0, 1)

        ax = axes[j, i]
        ax.imshow(rgb.numpy(), origin="upper")
        ax.set_title(f"{n_walls} wall(s) — sample {j+1}")

        # Mark start (green circle) and target (red star)
        start = info["dot_position"].cpu().numpy()
        target = info["target_position"].cpu().numpy()
        ax.plot(start[0], start[1], "go", markersize=8, label="start")
        ax.plot(target[0], target[1], "r*", markersize=12, label="target")
        if i == 0 and j == 0:
            ax.legend(fontsize=8)

plt.suptitle("Multi-Wall Environment Visualization", fontsize=14)
plt.tight_layout()
plt.savefig("multi_wall_preview.png", dpi=150)
print("Saved to multi_wall_preview.png")
