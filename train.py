import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from SAM2UNet import SAM2UNet
from TSAMNet import TSAMNet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time


def parse_args():
    parser = argparse.ArgumentParser("SAM2-UNet")
    parser.add_argument("--hiera_path", type=str, required=True,
                        help="path to the sam2 pretrained hiera")
    parser.add_argument("--train_image_path", type=str, required=True,
                        help="path to the image that used to train the model")
    parser.add_argument("--train_mask_path", type=str, required=True,
                        help="path to the mask file for training")
    parser.add_argument('--save_path', type=str, required=True,
                        help="path to save checkpoints and plots")
    parser.add_argument("--epoch", type=int, default=20,
                        help="training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)

    return parser.parse_args()
# def structure_loss(pred, mask):
#     weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
#     wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask)*weit).sum(dim=(2, 3))
#     union = ((pred + mask)*weit).sum(dim=(2, 3))
#     wiou = 1 - (inter + 1)/(union - inter+1)
#     return (wbce + wiou).mean()

def compute_shape_field(gt_mask):
    """根据GT计算法向场 s_gt"""
    gx = F.conv2d(
        gt_mask,
        torch.tensor(
            [[[[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]]]],
            device=gt_mask.device,
            dtype=gt_mask.dtype
        ),
        padding=1
    )
    gy = F.conv2d(
        gt_mask,
        torch.tensor(
            [[[[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]]]],
            device=gt_mask.device,
            dtype=gt_mask.dtype
        ),
        padding=1
    )
    s = torch.cat([gx, gy], dim=1)
    s = F.normalize(s, dim=1)
    return s
class CCSShapeLoss(nn.Module):
    def __init__(self, lambda_shape=0.05):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_shape = lambda_shape

        self.sobel_x = torch.tensor(
            [[[[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]]]],
            dtype=torch.float32
        )
        self.sobel_y = torch.tensor(
            [[[[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]]]],
            dtype=torch.float32
        )

    def forward(self, logits, mask):
        L_bce = self.bce(logits, mask)

        prob = torch.sigmoid(logits)

        sobel_x = self.sobel_x.to(prob.device)
        sobel_y = self.sobel_y.to(prob.device)

        gx = F.conv2d(prob, sobel_x, padding=1)
        gy = F.conv2d(prob, sobel_y, padding=1)

        grad = torch.cat([gx, gy], dim=1)
        grad = F.normalize(grad, dim=1, eps=1e-6)

        s_gt = compute_shape_field(mask)
        s_gt = F.normalize(s_gt, dim=1, eps=1e-6)

        L_shape = F.relu(-(grad * s_gt).sum(1)).mean()

        return L_bce + self.lambda_shape * L_shape


def main():
    args = parse_args()
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda")
    model = TSAMNet(args.hiera_path).to(device)

    #optim = opt.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    optim = opt.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    os.makedirs(args.save_path, exist_ok=True)

    # 初始化损失记录
    epoch_losses = []
    total_start = time.time()

    for epoch in range(args.epoch):
        epoch_start = time.time()
        epoch_loss = 0.0
        model.train()
        criterion = CCSShapeLoss().to(device)
        for batch_idx, batch in enumerate(dataloader, 1):
            x = batch['image'].to(device)
            target = batch['label'].to(device)
            optim.zero_grad()

            # loss0 = structure_loss(pred0, target)
            # loss1 = structure_loss(pred1, target)
            # loss2 = structure_loss(pred2, target)

            logits, _, side1, side2 = model(x)
            loss0 = criterion(logits, target)
            loss1 = criterion(side1, target)
            loss2 = criterion(side2, target)

            loss = loss0 + 0.5*loss1 + 0.5*loss2
            loss.backward()
            optim.step()
            # 直接累加各batch loss
            epoch_loss += loss.item()

            # 保持原有batch loss输出逻辑
            if batch_idx % 50 == 0:
                print(f"[Batch] Epoch:{epoch + 1}/{args.epoch} "
                      f"Batch:{batch_idx}/{len(dataloader)} "
                      f"Loss:{loss.item():.4f}")

        # 计算并记录epoch平均loss
        avg_loss = epoch_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        # 保存损失曲线
        plot_loss(epoch_losses, args.save_path)
        #输出时间
        epoch_time = time.time() - epoch_start
        print(f"[Time] Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")
        # 保存模型检查点


        # if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epoch:
        #     ckpt_path = os.path.join(args.save_path, f'SAM2-UNet-epoch{epoch + 1}.pth')
        #     torch.save(model.state_dict(), ckpt_path)
        #     print(f"[Checkpoint] Saved to {ckpt_path}")


    total_time = time.time() - total_start
    print(f"\n[Total Time] All epochs completed in {total_time / 60:.2f} minutes")
    print(f"[Average Time] {total_time / args.epoch:.2f} seconds per epoch")  # 新增此行

def plot_loss(loss_list, save_dir):#画图
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, 'b-o', linewidth=2)
    plt.title("Training Loss Curve", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True)

    save_path = Path(save_dir) / "training_loss.png"
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

# def seed_torch(seed=1024):
# 	random.seed(seed)
# 	os.environ['PYTHONHASHSEED'] = str(seed)
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed)
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # seed_torch(1024)
    main()