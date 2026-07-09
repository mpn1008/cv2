from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class FixationDataset(Dataset):
    """Pairs scene images with their corresponding fixation density maps."""

    # ImageNet stats for normalising the scene image going into the pretrained encoder
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(self, data_root: Path, images_txt: str, fixations_txt: str, image_size: int = 224):
        self.data_root = Path(data_root)

        image_paths = (self.data_root / images_txt).read_text().strip().splitlines()
        fix_paths = (self.data_root / fixations_txt).read_text().strip().splitlines()
        assert len(image_paths) == len(fix_paths), "image/fixation list length mismatch"

        self.image_paths = [self.data_root / p for p in image_paths]
        self.fix_paths = [self.data_root / p for p in fix_paths]

        self.scene_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self._MEAN, std=self._STD),
            ]
        )
        self.fix_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),  # → [0, 1] float32, shape (1, H, W)
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        scene = Image.open(self.image_paths[idx]).convert("RGB")
        fixation = Image.open(self.fix_paths[idx]).convert("L")
        return self.scene_tf(scene), self.fix_tf(fixation)


def build_loaders(
    train_root: Path,
    val_root: Path,
    train_images_txt: str = "train_images.txt",
    train_fixations_txt: str = "train_fixations.txt",
    val_images_txt: str = "val_images.txt",
    val_fixations_txt: str = "val_fixations.txt",
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    train_ds = FixationDataset(train_root, train_images_txt, train_fixations_txt, image_size)
    val_ds = FixationDataset(val_root, val_images_txt, val_fixations_txt, image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
