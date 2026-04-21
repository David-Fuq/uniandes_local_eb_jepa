import numpy as np
import torch

from eb_jepa.datasets.two_rooms.env import DotWall
from eb_jepa.datasets.two_rooms.utils import generate_multi_wall_layouts
from eb_jepa.datasets.two_rooms.wall_dataset import WallDataset, WallDatasetConfig


def _make_config(**overrides):
    defaults = dict(
        img_size=65,
        wall_padding=20,
        door_padding=10,
        wall_width=3,
        door_space=4,
        border_wall_loc=5,
        fix_wall=True,
        fix_wall_location=32,
        fix_door_location=18,
        n_walls=1,
        min_wall_spacing=12,
        device="cpu",
    )
    defaults.update(overrides)
    return WallDatasetConfig(**defaults)


class TestMultiWallLayoutGeneration:
    def test_generates_correct_number_of_walls(self):
        config = _make_config(n_walls=3)
        rng = np.random.default_rng(42)
        layouts = generate_multi_wall_layouts(config, rng=rng)
        assert len(layouts) == 3

    def test_walls_are_sorted_by_position(self):
        config = _make_config(n_walls=2)
        rng = np.random.default_rng(42)
        layouts = generate_multi_wall_layouts(config, rng=rng)
        positions = [w["wall_pos"] for w in layouts]
        assert positions == sorted(positions)

    def test_walls_respect_minimum_spacing(self):
        config = _make_config(n_walls=3, min_wall_spacing=12)
        rng = np.random.default_rng(42)
        for _ in range(50):
            layouts = generate_multi_wall_layouts(config, rng=rng)
            positions = [w["wall_pos"] for w in layouts]
            for i in range(len(positions) - 1):
                assert positions[i + 1] - positions[i] >= config.min_wall_spacing - 1

    def test_walls_within_valid_range(self):
        config = _make_config(n_walls=3)
        rng = np.random.default_rng(42)
        layouts = generate_multi_wall_layouts(config, rng=rng)
        for w in layouts:
            assert w["wall_pos"] >= config.wall_padding
            assert w["wall_pos"] <= config.img_size - config.wall_padding

    def test_doors_within_valid_range(self):
        config = _make_config(n_walls=3)
        rng = np.random.default_rng(42)
        layouts = generate_multi_wall_layouts(config, rng=rng)
        for w in layouts:
            assert w["door_pos"] >= config.door_padding
            assert w["door_pos"] <= config.img_size - config.door_padding

    def test_too_many_walls_raises(self):
        config = _make_config(n_walls=10, min_wall_spacing=12)
        rng = np.random.default_rng(42)
        try:
            generate_multi_wall_layouts(config, rng=rng)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestMultiWallEnvRendering:
    def test_single_wall_unchanged(self):
        config = _make_config(n_walls=1)
        env = DotWall(config, rng=np.random.default_rng(42))
        obs, info = env.reset()
        assert obs.shape == (2, 65, 65)
        wall_channel = obs[1]
        wall_pixels = (wall_channel > 0).sum().item()
        assert wall_pixels > 0

    def test_multi_wall_renders_multiple_stripes(self):
        config = _make_config(n_walls=3, fix_wall=False)
        env = DotWall(config, rng=np.random.default_rng(42))
        obs, info = env.reset()
        assert obs.shape == (2, 65, 65)
        wall_channel = obs[1].float()
        assert len(env.walls) == 3
        for wall_x, _ in env.walls:
            col = int(wall_x.item())
            if 0 <= col < 65:
                assert wall_channel[col, :].sum() > 0


class TestMultiWallEnvCollision:
    def test_cannot_cross_wall_except_at_door(self):
        config = _make_config(n_walls=2, fix_wall=False)
        env = DotWall(
            config, rng=np.random.default_rng(42), n_steps=200, n_allowed_steps=200
        )
        env.reset()

        sorted_walls = sorted(env.walls, key=lambda w: w[0].item())
        wall_x = sorted_walls[0][0]
        half_w = config.wall_width // 2

        env.dot_position = torch.tensor(
            [wall_x.item() - half_w - 2.0, 10.0], device=env.device
        )

        action = torch.tensor([5.0, 0.0])
        new_pos = env._calculate_next_position(action)
        assert new_pos[0].item() < wall_x.item() + half_w + 1

    def test_can_cross_wall_at_door(self):
        config = _make_config(n_walls=2, fix_wall=False)
        env = DotWall(
            config, rng=np.random.default_rng(42), n_steps=200, n_allowed_steps=200
        )
        env.reset()

        sorted_walls = sorted(env.walls, key=lambda w: w[0].item())
        wall_x = sorted_walls[0][0]
        door_y = sorted_walls[0][1]
        half_w = config.wall_width // 2

        env.dot_position = torch.tensor(
            [wall_x.item() - half_w - 1.0, door_y.item()], device=env.device
        )

        action = torch.tensor([half_w * 2 + 2.0, 0.0])
        new_pos = env._calculate_next_position(action)
        assert new_pos[0].item() > wall_x.item()


class TestMultiWallDataset:
    def test_single_wall_dataset_unchanged(self):
        config = _make_config(
            n_walls=1, size=10, val_size=10, n_steps=25, sample_length=17
        )
        ds = WallDataset(config)
        sample = ds[0]
        assert sample.states.shape[1] == config.sample_length

    def test_multi_wall_dataset_generates_samples(self):
        config = _make_config(
            n_walls=2,
            fix_wall=False,
            size=10,
            val_size=10,
            n_steps=25,
            sample_length=17,
            cross_wall_rate=0.0,
            expert_cross_wall_rate=0.0,
            wall_bump_rate=0.0,
        )
        ds = WallDataset(config)
        sample = ds[0]
        assert sample.states.shape[1] == config.sample_length
        assert sample.states.shape[0] == 2  # channels: dot + wall
