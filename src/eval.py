"""Evaluate a trained checkpoint (or bare pretrained ResNet-50) on the validation set.
 
Usage:
    # Pretrained ResNet-50 baseline (no checkpoint):
    python -m src.eval --val-root data/validation
 
    # Trained checkpoint:
    python -m src.eval --checkpoint checkpoints/resnet50_20250712_120000/best.pt \
                        --val-root data/validation
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.data import FixationDataset
from .losses import SaliencyLoss, compute_metrics
from .models import SaliencyNet


def evaluate(
    model: SaliencyNet,
    loader: DataLoader,
    criterion: SaliencyLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_totals: dict[str, float] = {}
    metric_totals: dict[str, float] = {}

    with torch.no_grad():
        for scene, fixation in tqdm(loader, desc="eval"):
            scene = scene.to(device, non_blocking=True)
            fixation = fixation.to(device, non_blocking=True)

            pred = model(scene)
            _, breakdown = criterion(pred, fixation)
            metrics = compute_metrics(pred, fixation)

            for k, v in breakdown.items():
                loss_totals[k] = loss_totals.get(k, 0.0) + v
            for k, v in metrics.items():
                metric_totals[k] = metric_totals.get(k, 0.0) + v

    n = len(loader)
    results = {k: v / n for k, v in loss_totals.items()}
    results.update({k: v / n for k, v in metric_totals.items()})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixation model on validation set")
    parser.add_argument("--val-root", type=Path, default=Path("data/validation"))
    parser.add_argument("--val-images-txt", default="val_images.txt")
    parser.add_argument("--val-fixations-txt", default="val_fixations.txt")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to best.pt checkpoint. Omit to run bare pretrained ResNet-50 baseline.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        cfg = ckpt.get("cfg", None)
        decoder_channels = cfg.model.decoder_channels if cfg else None
        model = SaliencyNet(pretrained=False, decoder_channels=decoder_channels)
        model.load_state_dict(ckpt["model"])
        print(f"  Epoch: {ckpt.get('epoch', '?')}")
    else:
        print("No checkpoint given – running pretrained ResNet-50 baseline (no saliency training)")
        model = SaliencyNet(pretrained=True)

    model = model.to(device)

    # Data
    val_ds = FixationDataset(
        args.val_root, args.val_images_txt, args.val_fixations_txt, image_size=args.image_size
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Val samples: {len(val_ds)} | batches: {len(val_loader)}")

    criterion = SaliencyLoss()
    results = evaluate(model, val_loader, criterion, device)

    print("\n── Validation Results ──────────────────")
    print(f"  Val Loss  : {results.get('loss/total', float('nan')):.4f}")
    print(f"  KL        : {results.get('loss/kl', float('nan')):.4f}")
    print(f"  CC        : {results.get('metric/cc', float('nan')):.4f}")
    print(f"  AUC-Judd  : {results.get('metric/auc_judd', float('nan')):.4f}")
    print("────────────────────────────────────────")


if __name__ == "__main__":
    main()
