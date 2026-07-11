"""Run inference on test images and save predicted fixation maps.
 
Usage (from repo root):
    python3 -m src.inference \\
        --checkpoint checkpoints/resnet50_20260709_173527/best.pt \\
        --test-root  data/testing \\
        --test-txt   test_images.txt \\
        --output-dir predictions
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .models import SaliencyNet


# ImageNet normalisation used during training
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[SaliencyNet, int]:
    """Load SaliencyNet from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = ckpt.get("cfg")
    if cfg is not None:
        decoder_channels = cfg.model.decoder_channels
    else:
        # Fallback for manually saved state dicts
        decoder_channels = [256, 128, 64, 32]

    model = SaliencyNet(pretrained=False, decoder_channels=decoder_channels)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    epoch = ckpt.get("epoch", -1)
    return model, epoch


@torch.no_grad()
def predict(
    checkpoint: Path,
    test_root: Path,
    test_txt: str,
    output_dir: Path,
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> None:
    model, epoch = load_model(checkpoint, device)
    print(f"Loaded checkpoint (epoch {epoch}) — backbone: {model.encoder.__class__.__name__}")

    # Read image list
    image_paths = [
        test_root / p.strip() for p in (test_root / test_txt).read_text().strip().splitlines()
    ]
    print(f"Test images: {len(image_paths)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    scene_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ]
    )

    # Process in batches
    for start in tqdm(range(0, len(image_paths), batch_size), desc="predicting"):
        batch_paths = image_paths[start : start + batch_size]
        originals = [Image.open(p).convert("RGB") for p in batch_paths]
        orig_sizes = [img.size for img in originals]  # (W, H)

        batch = torch.stack([scene_tf(img) for img in originals]).to(device)
        preds = model(batch)  # (B, 1, H, W) in [0, 1]

        for pred, path, (orig_w, orig_h) in zip(preds, batch_paths, orig_sizes):
            # Resize prediction back to original image dimensions
            pred_img = transforms.functional.resize(
                pred.cpu(),
                [orig_h, orig_w],
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )
            # Convert to uint8 grayscale PIL image
            pred_np = (pred_img.squeeze(0).numpy() * 255).clip(0, 255).astype("uint8")
            out_name = f"pred_{path.stem}.png"
            Image.fromarray(pred_np, mode="L").save(output_dir / out_name)

    print(f"Saved {len(image_paths)} predictions → {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fixation prediction on test images")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a training checkpoint (.pt)",
    )
    parser.add_argument("--test-root", type=Path, default="data/testing")
    parser.add_argument("--test-txt", type=str, default="test_images.txt")
    parser.add_argument("--output-dir", type=Path, default="predictions")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    predict(
        checkpoint=args.checkpoint,
        test_root=args.test_root,
        test_txt=args.test_txt,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=device,
    )
