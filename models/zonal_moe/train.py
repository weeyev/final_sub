import os
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from torch.backends import cudnn

from .model import ZonalMoE
from .preprocessing import (
    AirfoilDataset,
    geometry_aware_split,
    compute_dataset_statistics,
    compute_polynomial_baseline,
)


class WeightedMSELoss(nn.Module):
    def __init__(self, wall_threshold: float, turbulent_weight: float = 2.0):
        super().__init__()
        self.wall_threshold = wall_threshold
        self.turbulent_weight = turbulent_weight

    def forward(self, pred_residual, target_residual, wall_dist):
        near_wall = wall_dist <= self.wall_threshold
        zone_weight = torch.ones_like(wall_dist)
        zone_weight[near_wall] = self.turbulent_weight
        weight = zone_weight.unsqueeze(0).unsqueeze(-1)
        diff = pred_residual - target_residual
        loss = (weight * diff.pow(2)).mean()
        return loss


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    grad_accum_steps=4,
    scaler=None,
    use_amp=True,
):
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        vel_in = batch["vel_in"].squeeze(0).to(device, non_blocking=True)
        vel_out = batch["vel_out"].squeeze(0).to(device, non_blocking=True)
        wall_dist = batch["wall_dist"].squeeze(0).to(device, non_blocking=True)
        is_airfoil = batch["is_airfoil"].squeeze(0).to(device, non_blocking=True)
        edge_index = batch["edge_index"].squeeze(0).to(device, non_blocking=True)
        edge_index_dense = (
            batch["edge_index_dense"].squeeze(0).to(device, non_blocking=True)
        )

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(use_amp and scaler is not None)):
            residual_pred = model(
                vel_in=vel_in,
                edge_index=edge_index,
                wall_dist=wall_dist,
                is_airfoil=is_airfoil,
                edge_index_dense=edge_index_dense,
            )
            poly_baseline = compute_polynomial_baseline(vel_in)
            target_residual = vel_out - poly_baseline
            main_loss = criterion(residual_pred, target_residual, wall_dist)
            balance_loss = getattr(model, "balance_loss", 0.0)
            loss = (main_loss + balance_loss) / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1

        if (batch_idx + 1) % 10 == 0:
            with torch.no_grad():
                gate_stats = model.get_routing_stats()
            main_loss_val = main_loss.item()
            bal_loss_val = getattr(model, "balance_loss", 0.0)
            if isinstance(bal_loss_val, torch.Tensor):
                bal_loss_val = bal_loss_val.item()
            print(
                f"  batch {batch_idx + 1}/{len(dataloader)} | loss: {main_loss_val:.6f} (bal: {bal_loss_val:.6f}) | "
                f"gate: {gate_stats['gate_mean']:.3f}±{gate_stats['gate_std']:.3f} | "
                f"turb: {gate_stats['turbulent_fraction'] * 100:.1f}%"
            )

    if num_batches % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / num_batches


def validate(model, dataloader, criterion, device, use_amp=True):
    """Validate model on held-out geometries."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            vel_in = batch["vel_in"].squeeze(0).to(device, non_blocking=True)
            vel_out = batch["vel_out"].squeeze(0).to(device, non_blocking=True)
            wall_dist = batch["wall_dist"].squeeze(0).to(device, non_blocking=True)
            is_airfoil = batch["is_airfoil"].squeeze(0).to(device, non_blocking=True)
            edge_index = batch["edge_index"].squeeze(0).to(device, non_blocking=True)
            edge_index_dense = (
                batch["edge_index_dense"].squeeze(0).to(device, non_blocking=True)
            )

            # Model returns RESIDUAL
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                residual_pred = model(
                    vel_in=vel_in,
                    edge_index=edge_index,
                    wall_dist=wall_dist,
                    is_airfoil=is_airfoil,
                    edge_index_dense=edge_index_dense,
                )

            # Polynomial baseline
            poly_baseline = compute_polynomial_baseline(vel_in)
            target_residual = vel_out - poly_baseline
            loss = criterion(residual_pred, target_residual, wall_dist)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """Save training checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
        },
        path,
    )
    print(f"Saved checkpoint to {path}")


def load_checkpoint(model, optimizer, scheduler, path, device):
    """Load training checkpoint.

    Note: weights_only=False is intentional — the checkpoint includes optimizer
    and scheduler state dicts which require full unpickling.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]


def train(
    data_dir: str,
    output_dir: str = "./checkpoints",
    epochs: int = 100,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    grad_accum_steps: int = 4,
    turbulent_weight: float = 2.0,
    device: str = None,
    resume: str = None,
    seed: int = 42,
    use_amp: bool = True,
):

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device == "cuda" and torch.cuda.is_available():
        cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"cuDNN benchmark enabled, float32 matmul precision: high")

    print(f"Loading data from {data_dir}")
    dataset = AirfoilDataset(
        data_dir=data_dir,
        k_standard=16,
        k_dense=32,
        use_cache=True,
        normalize=True,
        stats_sample_size=30,
    )
    print(f"Found {len(dataset)} samples")

    # Compute statistics for model initialization
    stats = compute_dataset_statistics(dataset)
    print(
        f"Dataset stats: wall_dist_p95={stats['wall_dist_scale']:.3f}, "
        f"vorticity_p95={stats['vorticity_scale']:.3f}, "
        f"wall_dist_threshold={stats['wall_dist_threshold']:.3f}"
    )

    train_indices, val_indices = geometry_aware_split(dataset, val_ratio=0.1, seed=seed)
    print(f"Split: {len(train_indices)} train, {len(val_indices)} val samples")

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )

    model = ZonalMoE(
        d_temporal=64,
        backbone_hidden=128,
        laminar_hidden=64,
        turbulent_hidden=256,
        heads=4,
        dropout=0.1,
        output_timesteps=5,
        output_channels=3,
        wall_dist_scale=stats["wall_dist_scale"],
        vorticity_scale=stats["vorticity_scale"],
    ).to(device)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Loss uses fixed threshold from dataset
    criterion = WeightedMSELoss(
        wall_threshold=stats["wall_dist_threshold"], turbulent_weight=turbulent_weight
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    start_epoch = 0
    best_val_loss = float("inf")

    if resume is not None:
        print(f"Resuming from {resume}")
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, resume, device)
        start_epoch += 1

    print(f"\nStarting training for {epochs} epochs")
    print(f"Gradient accumulation: {grad_accum_steps} steps")
    print(
        f"Turbulent weight: {turbulent_weight}x for wall_dist < {stats['wall_dist_threshold']:.3f}"
    )
    print("-" * 60)

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_accum_steps,
            scaler,
            use_amp,
        )

        val_loss = validate(model, val_loader, criterion, device, use_amp)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d} | train: {train_loss:.6f} | val: {val_loss:.6f} | "
            f"lr: {current_lr:.2e} | time: {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                output_dir / "best_model.pt",
            )

        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                output_dir / f"checkpoint_epoch{epoch:03d}.pt",
            )

    print("-" * 60)
    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    print(f"Best model saved to {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Zonal MoE for airflow prediction"
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints", help="Output directory"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--grad_accum", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--turbulent_weight", type=float, default=2.0, help="Turbulent zone weight"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no_amp",
        action="store_true",
        default=False,
        help="Disable automatic mixed precision",
    )
    parser.add_argument("--cpu", action="store_true", help="Debug: force CPU")

    args = parser.parse_args()

    # Force CPU + disable optimizations if --cpu flag
    if args.cpu:
        args.device = "cpu"
        args.no_amp = True

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_accum_steps=args.grad_accum,
        turbulent_weight=args.turbulent_weight,
        device=args.device,
        resume=args.resume,
        seed=args.seed,
        use_amp=not args.no_amp,
    )
