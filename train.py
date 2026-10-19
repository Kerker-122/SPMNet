"""Three-scale ICCS training with separately supervised shape heads."""
import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import FullDataset
from SPMNet import SPMNet
from ccs import ShapeFieldLoss
from losses import HybridShapeAwareLoss


def parse_args():
    parser = argparse.ArgumentParser("SPMNet multi-scale ICCS")
    parser.add_argument("--hiera_path", required=True)
    parser.add_argument("--train_image_path", required=True)
    parser.add_argument("--train_mask_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--size", type=int, default=352)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--iccs_iterations", type=int, default=10)
    parser.add_argument("--tau_q", type=float, default=1.0)
    parser.add_argument("--xi", type=float, default=1.0)
    parser.add_argument("--uncertainty_beta", type=float, default=2.0)
    parser.add_argument("--field_epsilon", type=float, default=0.01,
                        help="Softmin temperature in normalized image-length units")
    parser.add_argument("--field_points", type=int, default=30)
    parser.add_argument("--lambda_shape", type=float, default=0.05,
                        help="Original Sobel boundary-direction consistency weight in HSAL")
    parser.add_argument("--lambda_field", type=float, default=0.1,
                        help="Separate auxiliary Shape Head supervision weight")
    parser.add_argument("--scale_weights", type=float, nargs=3, default=(0.5, 0.5, 1.0),
                        help="Coarse, middle, fine supervision weights")
    parser.add_argument("--joint_field_training", action="store_true",
                        help="Also allow segmentation gradients into Shape Heads")
    parser.add_argument("--disable_cross_scale_feedback", action="store_true",
                        help="Ablation: bypass refined-probability feedback between stages")
    args = parser.parse_args()
    if args.epoch < 1 or args.batch_size < 1 or args.save_every < 1:
        parser.error("epoch, batch_size and save_every must be positive")
    if args.field_epsilon <= 0 or args.field_points < 1:
        parser.error("field_epsilon and field_points must be positive")
    if args.lambda_field <= 0:
        parser.error("Independent Shape Head supervision requires lambda_field>0")
    return args


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run training on your CUDA server; no local CPU fallback")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "training_config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8")
    dataset = FullDataset(args.train_image_path, args.train_mask_path, args.size,
                          mode="train", with_shape_fields=True,
                          field_epsilon=args.field_epsilon, field_points=args.field_points)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(args.seed))
    device = torch.device("cuda")
    model = SPMNet(args.hiera_path, iccs_iterations=args.iccs_iterations,
                    tau_q=args.tau_q, xi=args.xi, uncertainty_beta=args.uncertainty_beta,
                    detach_field=not args.joint_field_training,
                    use_cross_scale_feedback=not args.disable_cross_scale_feedback).to(device)
    criterion = HybridShapeAwareLoss(args.lambda_shape, args.scale_weights).to(device)
    field_criterion = ShapeFieldLoss(args.scale_weights).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=1e-7)
    losses = []
    start = time.time()
    with (save_dir / "loss_history.csv").open("w", newline="", encoding="utf-8") as log:
        writer = csv.DictWriter(log, fieldnames=("epoch", "lr", "total", "hsal", "bce", "shape", "field"))
        writer.writeheader()
        for epoch in range(args.epoch):
            model.train()
            totals = {key: 0.0 for key in ("total", "hsal", "bce", "shape", "field")}
            count = 0
            epoch_lr = optimizer.param_groups[0]["lr"]
            for batch_idx, batch in enumerate(loader, 1):
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["label"].to(device, non_blocking=True)
                fields = [s.to(device, non_blocking=True) for s in batch["shape_fields"]]
                valid = [v.to(device, non_blocking=True) for v in batch["shape_valid"]]
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                hsal, parts = criterion(outputs, masks)
                field_loss = field_criterion(outputs, fields, valid)
                loss = hsal + args.lambda_field * field_loss
                parts.update(total=loss, field=field_loss)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch+1}, batch {batch_idx}")
                loss.backward()
                optimizer.step()
                n = images.shape[0]
                count += n
                for key in totals:
                    totals[key] += parts[key].detach().item() * n
                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch+1}/{args.epoch}, batch {batch_idx}/{len(loader)}, "
                          f"loss={loss.item():.6f}")
            averages = {key: value / count for key, value in totals.items()}
            writer.writerow(dict(epoch=epoch+1, lr=epoch_lr, **averages))
            log.flush()
            losses.append(averages["total"])
            scheduler.step()
            print(f"Epoch {epoch+1}: {averages}")
            if (epoch+1) % args.save_every == 0 or epoch+1 == args.epoch:
                torch.save(dict(model=model.state_dict(), model_config=model.model_config,
                                training_config=vars(args), epoch=epoch+1,
                                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()),
                           save_dir / f"SPMNet-epoch{epoch+1}.pth")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses)+1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.grid(True)
    plt.savefig(save_dir / "training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training completed in {(time.time()-start)/60:.2f} minutes")


if __name__ == "__main__":
    main()
