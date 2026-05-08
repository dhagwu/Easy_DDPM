"""
InSAR patch dataset — loads .npz files and returns (condition, target) pairs.
"""
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class InSARDataset(Dataset):
    """Loads .npz patches, returns wrapped-phase conditioning + unwrapped target."""

    def __init__(self, data_dir, split="train", image_size=128, num_samples=None,
                 use_sincos=True, source_domain=None):
        self.image_size = image_size
        self.use_sincos = use_sincos
        self.split = split

        file_dir = os.path.join(data_dir, split)
        all_files = sorted(glob.glob(os.path.join(file_dir, "*.npz")))

        # filter by source_domain if specified
        if source_domain is not None:
            filtered = []
            for f in all_files:
                try:
                    d = np.load(f, allow_pickle=False)
                    if str(d["source_domain"]) == source_domain:
                        filtered.append(f)
                except Exception:
                    pass
            all_files = filtered

        if num_samples is not None and len(all_files) > num_samples:
            random.seed(42)
            all_files = sorted(random.sample(all_files, num_samples))

        self.files = all_files
        # cache metadata for fast lookup
        self._meta_cache = {}

    def __len__(self):
        return len(self.files)

    def _load_meta(self, idx):
        """Load scalar metadata for a sample (cached)."""
        if idx not in self._meta_cache:
            data = np.load(self.files[idx], allow_pickle=False)
            self._meta_cache[idx] = {
                "source_domain": str(data["source_domain"]),
                "scenario_name": str(data["scenario_name"]),
                "deformation_type": str(data["deformation_type"]),
                "noise_level": str(data["noise_level"]),
                "coherence_level": str(data["coherence_level"]),
                "gradient_level": str(data["gradient_level"]),
                "pair_id": str(data["pair_id"]),
                "difficulty_score": float(data["difficulty_score"]),
                "valid_ratio": float(data["valid_ratio"]),
            }
        return self._meta_cache[idx]

    def get_meta(self, idx):
        """Return group labels for evaluation."""
        return self._load_meta(idx)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        wrapped = data["wrapped"].astype(np.float32)
        unwrapped = data["unwrapped"].astype(np.float32)
        coherence = data["coherence"].astype(np.float32)
        mask = data["mask"].astype(np.float32)

        h, w = wrapped.shape
        if self.split == "train":
            top = random.randint(0, h - self.image_size)
            left = random.randint(0, w - self.image_size)
        else:
            top = (h - self.image_size) // 2
            left = (w - self.image_size) // 2

        wrapped = wrapped[top:top + self.image_size, left:left + self.image_size]
        unwrapped = unwrapped[top:top + self.image_size, left:left + self.image_size]
        coherence = coherence[top:top + self.image_size, left:left + self.image_size]
        mask = mask[top:top + self.image_size, left:left + self.image_size]

        if self.use_sincos:
            condition = np.stack([np.sin(wrapped), np.cos(wrapped), coherence], axis=0)
        else:
            condition = np.stack([wrapped, coherence], axis=0)

        target = unwrapped[np.newaxis, ...]

        # cache meta on first load
        if idx not in self._meta_cache:
            self._meta_cache[idx] = {
                "source_domain": str(data["source_domain"]),
                "scenario_name": str(data["scenario_name"]),
                "deformation_type": str(data["deformation_type"]),
                "noise_level": str(data["noise_level"]),
                "coherence_level": str(data["coherence_level"]),
                "gradient_level": str(data["gradient_level"]),
                "pair_id": str(data["pair_id"]),
                "difficulty_score": float(data["difficulty_score"]),
                "valid_ratio": float(data["valid_ratio"]),
            }

        return torch.from_numpy(condition), torch.from_numpy(target)


def make_dataloaders(config):
    """Create train/val dataloaders."""
    ds_train = InSARDataset(
        config.DATA_DIR, split="train",
        image_size=config.IMAGE_SIZE,
        num_samples=config.NUM_TRAIN,
        use_sincos=config.USE_SINCOS,
    )
    ds_val = InSARDataset(
        config.DATA_DIR, split="val",
        image_size=config.IMAGE_SIZE,
        num_samples=config.NUM_VAL,
        use_sincos=config.USE_SINCOS,
    )
    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    return dl_train, dl_val


def make_test_dataset(config):
    """Create test dataset (for evaluation)."""
    return InSARDataset(
        config.DATA_DIR, split="val",
        image_size=config.IMAGE_SIZE,
        num_samples=config.NUM_TEST,
        use_sincos=config.USE_SINCOS,
    )
