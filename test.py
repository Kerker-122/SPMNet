import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from dataset import TestDataset
import matplotlib.pyplot as plt
from pathlib import Path
from TSAMNet import TSAMNet



parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                help="path to the checkpoint of sam2-unet")
parser.add_argument("--test_image_path", type=str, required=True,
                    help="path to the image files for testing")
parser.add_argument("--test_gt_path", type=str, required=True,
                    help="path to the mask files for testing")
parser.add_argument("--save_path", type=str, required=True,
                    help="path to save the predicted masks")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = TestDataset(args.test_image_path, args.test_gt_path, 352)

model = TSAMNet().to(device)

model.load_state_dict(torch.load(args.checkpoint), strict=False)
model.eval()
model.cuda()
os.makedirs(args.save_path, exist_ok=True)
for i in range(test_loader.size):
    with torch.no_grad():
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        image = image.to(device)

        logits, _, _, _ = model(image)
        res = torch.sigmoid(logits)

        # res, _, _ = model(image)
        # print(
        #     "logits stats:",
        #     res.min().item(),
        #     res.max().item(),
        #     res.mean().item(),
        #     flush = True
        # )
        # break

        # fix: duplicate sigmoid
        # res = torch.sigmoid(res)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)

        #res = res.sigmoid().data.cpu()
        res = res.data.cpu()

        res = res.numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)
        # If you want to binarize the prediction Ablation, please uncomment the following three lines.
        # Note that this action will affect the calculation of evaluation metrics.
        # lambda = 0.5
        # res[res >= int(255 * lambda)] = 255
        # res[res < int(255 * lambda)] = 0
        print("Saving " + name)
        imageio.imsave(os.path.join(args.save_path, name[:-4] + ".png"), res)

