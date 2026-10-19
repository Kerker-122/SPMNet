import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from ccs import CCSSigmoid

class MonaAdaper(nn.Module):
    """
    CVPR 2025 Mona Adapter
    Monotonic Nonlinear Adapter (single-class version)
    """
    def __init__(self, blk, bottleneck_dim=32):
        super().__init__()
        self.block = blk

        dim = blk.attn.qkv.in_features  # SAM2 ViT hidden dim

        # 冻结原始 Transformer Block
        for p in self.block.parameters():
            p.requires_grad = False

        # Bottleneck projection
        self.down = nn.Linear(dim, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, dim, bias=False)

        # === Mona Monotonic Nonlinearity (inline) ===
        self.mono_scale = nn.Parameter(torch.ones(bottleneck_dim))
        self.mono_bias = nn.Parameter(torch.zeros(bottleneck_dim))

        # Identity-aware initialization（论文关键）
        nn.init.zeros_(self.up.weight)

    def monotonic_nonlinear(self, x):
        """
        Monotonic nonlinear function
        Softplus guarantees positive derivative
        """
        return F.softplus(self.mono_scale * x + self.mono_bias)

    def forward(self, x):
        delta = self.up(
            self.monotonic_nonlinear(
                self.down(x)
            )
        )
        return self.block(x + delta)

class SAGAttention(nn.Module):
    def __init__(self, in_channels):
        super(SAGAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.conv1(x)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        return x * attn
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
        self.sag = SAGAttention(in_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.sag(x)
        x = self.conv(x)
        return x 


class CCSFeedbackFusion(nn.Module):
    """Feed a previous stage's refined probability into current features.

    F_next = F_fused + Conv3x3(GELU(Conv1x1([F_fused, up(P_ccs)]))).
    The residual feature path preserves information outside the coarse mask;
    probability feedback is differentiable and is not thresholded or detached.
    """
    def __init__(self, channels=64):
        super().__init__()
        self.project = nn.Conv2d(channels + 1, channels, kernel_size=1, bias=False)
        self.activation = nn.GELU()
        self.refine = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, features, previous_probability):
        guidance = F.interpolate(previous_probability, size=features.shape[-2:],
                                 mode="bilinear", align_corners=False)
        inputs = torch.cat((features, guidance.to(features.dtype)), dim=1)
        return features + self.refine(self.activation(self.project(inputs)))


class MultiScaleSpatialAttention(nn.Module):
    def __init__(self, in_channels, rate=4):
        super().__init__()
        self.conv3 = nn.Conv2d(in_channels, in_channels // rate, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels // rate, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, in_channels // rate, kernel_size=7, padding=3)

        # 动态权重融合
        self.fuse_weight = nn.Parameter(torch.tensor([0.33, 0.33, 0.34]), requires_grad=True)

        self.fuse_conv = nn.Conv2d(3 * (in_channels // rate), in_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x3 = F.relu(self.conv3(x))
        x5 = F.relu(self.conv5(x))
        x7 = F.relu(self.conv7(x))

        # 权重融合
        weighted = torch.cat([x3, x5, x7], dim=1)
        sp_attn = self.norm(self.fuse_conv(weighted)).sigmoid()
        return x * sp_attn
class TwinAttn(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(TwinAttn, self).__init__()
        self.in_channels = in_channels

        # 通道注意力子模块
        self.channel_attention = nn.Sequential(
            # 降维，减少参数数量和计算复杂度
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),  # 非线性激活
            # 升维，恢复到原始通道数
            nn.Linear(int(in_channels / rate), in_channels)
        )

        #空间注意力子模块
        self.spatial_attention = nn.Sequential(
            # 使用7x7卷积核进行空间特征的降维处理
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),  # 批归一化，加速收敛，提升稳定性
            nn.ReLU(inplace=True),  # 非线性激活
            # 使用7x7卷积核进行空间特征的升维处理
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels)  # 批归一化
        )
        #self.spatial_attention = MultiScaleSpatialAttention(in_channels, rate)#替换空间注意力


    def forward(self, x):
        b, c, h, w = x.shape  # 输入张量的维度信息
        # 调整张量形状以适配通道注意力处理
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)

        # 应用通道注意力，并恢复原始张量形状
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        # 生成通道注意力图
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()

        # 应用通道注意力图进行特征加权
        x = x * x_channel_att

        # 生成空间注意力图并应用进行特征加权
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# class Up(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#
#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)

class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        net = self.block(x + prompt)
        return net


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

# 示例使用
class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class SPMNet(nn.Module):
    def __init__(self, checkpoint_path=None, iccs_iterations=10, tau_q=1.0,
                 xi=1.0, uncertainty_beta=2.0, detach_field=True,
                 use_cross_scale_feedback=True) -> None:
        super(SPMNet, self).__init__()
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path, device="cpu")
        else:
            model = build_sam2(model_cfg, device="cpu")

        self.model_config = dict(iccs_iterations=iccs_iterations, tau_q=tau_q,
                                 xi=xi, uncertainty_beta=uncertainty_beta,
                                 detach_field=detach_field,
                                 use_cross_scale_feedback=use_cross_scale_feedback)
        self.detach_field = detach_field
        self.use_cross_scale_feedback = use_cross_scale_feedback

        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck

        self.encoder = model.image_encoder.trunk
        for param in self.encoder.parameters():
            param.requires_grad = False                  #将 SAM 的编码器主干设置为不可训练（仅用作特征提取器）


        blocks = []

        for block in self.encoder.blocks:
            blocks.append(MonaAdaper(block))                                            #在每个原始编码器块后插入 Adapter 层

        self.encoder.blocks = nn.Sequential(
            *blocks
        )
        self.rfb1 = RFB_modified(144, 64)
        self.rfb2 = RFB_modified(288, 64)
        self.rfb3 = RFB_modified(576, 64)
        self.rfb4 = RFB_modified(1152, 64)

        self.up1 = (Up(128, 64))
        self.up2 = (Up(128, 64))
        self.up3 = (Up(128, 64))
        # Both transitions receive the preceding CCS-refined prediction.
        self.feedback_middle = CCSFeedbackFusion(64)
        self.feedback_fine = CCSFeedbackFusion(64)

        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)

        self.head = nn.Conv2d(64, 1, kernel_size=1)
        self.gam1 = TwinAttn(144)
        self.gam2 = TwinAttn(288)
        self.gam3 = TwinAttn(576)
        self.gam4 = TwinAttn(1152)
        self.field_head = nn.Conv2d(64, 2, kernel_size=1)
        self.field_head_side1 = nn.Conv2d(64, 2, kernel_size=1)
        self.field_head_side2 = nn.Conv2d(64, 2, kernel_size=1)
        self.ccs_sigmoid = CCSSigmoid(iccs_iterations, tau_q, xi, uncertainty_beta)

    def decode_prediction(self, features, seg_head, field_head):
        raw_logits = seg_head(features)
        # Paper Fig. 3 uses Norm on the predicted field.
        field = F.normalize(field_head(features).float(), dim=1, eps=1e-6)
        # Default: Shape Heads receive their own cosine loss, not an indirect
        # segmentation-loss gradient. Shared decoder features still train jointly.
        solver_field = field.detach() if self.detach_field else field
        refined_logits, probability = self.ccs_sigmoid(raw_logits, solver_field)
        return dict(raw_logits=raw_logits, shape_field=field,
                    refined_logits=refined_logits, probability=probability)

    def forward(self, x):
        input_size = x.shape[-2:]
        x1, x2, x3, x4 = self.encoder(x)#[1, 144, 88, 88],[1, 288, 44, 44],[1, 576, 22, 22],[1, 1152, 11, 11]

        x1,x2,x3,x4=self.gam1(x1),self.gam2(x2),self.gam3(x3),self.gam4(x4)

        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)

        x = self.up1(x4, x3)
        coarse = self.decode_prediction(x, self.side1, self.field_head_side1)
        x = self.up2(x, x2)
        if self.use_cross_scale_feedback:
            x = self.feedback_middle(x, coarse["probability"])
        middle = self.decode_prediction(x, self.side2, self.field_head_side2)
        x = self.up3(x, x1)
        if self.use_cross_scale_feedback:
            x = self.feedback_fine(x, middle["probability"])
        fine = self.decode_prediction(x, self.head, self.field_head)
        refined_logits = F.interpolate(fine["refined_logits"], size=input_size,
                                       mode="bilinear", align_corners=False)
        return dict(
            stages=[coarse, middle, fine],
            refined_logits=refined_logits,
            out_ccs=torch.sigmoid(refined_logits),
            raw_logits=F.interpolate(fine["raw_logits"], size=input_size,
                                     mode="bilinear", align_corners=False),
        )

if __name__ == "__main__":
    with torch.no_grad():
        model = SPMNet().cuda()
        x = torch.randn(1, 3, 352, 352).cuda()
        outputs = model(x)
        print(outputs["out_ccs"].shape)
        print([stage["shape_field"].shape for stage in outputs["stages"]])

