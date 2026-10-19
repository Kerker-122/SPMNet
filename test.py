"""Save the finest ICCS prediction, using the checkpoint's solver settings."""
import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from dataset import TestDataset
from SPMNet import SPMNet
from checkpoint_utils import load_inference_model


def main():
    parser = argparse.ArgumentParser("SPMNet inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_image_path", required=True)
    parser.add_argument("--test_gt_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--iccs_iterations", type=int, default=None)
    parser.add_argument("--uncertainty_beta", type=float, default=None)
    parser.add_argument("--legacy_minmax", action="store_true",
                        help="Explicitly reproduce the previous per-image min-max protocol")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overrides = dict(iccs_iterations=args.iccs_iterations,
                     uncertainty_beta=args.uncertainty_beta)
    model, checkpoint = load_inference_model(SPMNet, args.checkpoint, device, overrides)
    size = args.size or checkpoint.get("training_config", {}).get("size", 352)
    loader = TestDataset(args.test_image_path, args.test_gt_path, size)
    os.makedirs(args.save_path, exist_ok=True)
    with (Path(args.save_path) / "inference_config.json").open("w", encoding="utf-8") as log:
        json.dump(dict(checkpoint=str(Path(args.checkpoint).resolve()), size=size,
                       model_config=model.model_config, legacy_minmax=args.legacy_minmax),
                  log, indent=2)
    with torch.inference_mode():
        for _ in range(loader.size):
            image, gt, name = loader.load_data()
            outputs = model(image.to(device))
            logits = F.interpolate(outputs["refined_logits"], size=gt.shape,
                                   mode="bilinear", align_corners=False)
            res = torch.sigmoid(logits)[0, 0].cpu().numpy()
            if args.legacy_minmax:
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = np.rint(np.clip(res, 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(Path(args.save_path) / (Path(name).stem + ".png"), res)


if __name__ == "__main__":
    main()

