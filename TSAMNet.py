import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2

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

class CCSSigmoid(nn.Module):
    def __init__(self, T=10, tau_q=1.0, xi=1.0):
        super(CCSSigmoid, self).__init__()
        self.T = T
        self.tau_q = tau_q
        self.xi = xi
        # 固定卷积核计算梯度与散度
        kx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ky = kx.transpose(2, 3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def gradient(self, u):
        gx = F.conv2d(u, self.kx, padding=1)
        gy = F.conv2d(u, self.ky, padding=1)
        return gx, gy

    def divergence(self, qx, qy):
        dx = F.conv2d(qx, -self.kx, padding=1)
        dy = F.conv2d(qy, -self.ky, padding=1)
        return dx + dy

    def forward(self, o, s):
        u = torch.sigmoid(o)
        qx = torch.zeros_like(o)
        qy = torch.zeros_like(o)
        # for _ in range(self.T):
        #     gx, gy = self.gradient(u)
        #     qx = F.relu(qx - self.tau_q * s[:, 0:1] * gx)
        #     qy = F.relu(qy - self.tau_q * s[:, 1:2] * gy)
        #     div_qs = self.divergence(qx * s[:, 0:1], qy * s[:, 1:2])
        #     u = torch.sigmoid((o - div_qs) / self.xi)
        # return u
        for _ in range(self.T):
            # >>> 1. 计算伪装不确定度 (Camouflage Uncertainty)
            # u * (1-u) 在 u=0.5 时最大，表示不确定；在 0 或 1 时最小。
            uncertainty = 4.0 * u * (1.0 - u)

            # >>> 2. 动态调节机制 (Dynamic Modulation)
            # 伪装区域 (uncertainty 大) -> 放大形状场 s 的作用 (1 + uncertainty)
            # 确定区域 (uncertainty 小) -> 保持原样
            # 这里的 alpha 是一个超参，可以设为 1.0 或 2.0
            cam_weight = 1.0 + 2.0 * uncertainty

            gx, gy = self.gradient(u)

            # 将加权项应用到通量计算中，强行用形状场 s 修正伪装区域
            qx = F.relu(qx - self.tau_q * s[:, 0:1] * gx * cam_weight)
            qy = F.relu(qy - self.tau_q * s[:, 1:2] * gy * cam_weight)

            div_qs = self.divergence(qx * s[:, 0:1], qy * s[:, 1:2])
            u = torch.sigmoid((o - div_qs) / self.xi)

        return u
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

class TSAMNet(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(TSAMNet, self).__init__()
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)

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
        self.up4 = (Up(128, 64))

        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)

        self.head = nn.Conv2d(64, 1, kernel_size=1)
        self.gam1 = TwinAttn(144)
        self.gam2 = TwinAttn(288)
        self.gam3 = TwinAttn(576)
        self.gam4 = TwinAttn(1152)
        self.field_head = nn.Conv2d(64, 2, kernel_size=1)
        self.ccs_sigmoid = CCSSigmoid(T=10)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)#[1, 144, 88, 88],[1, 288, 44, 44],[1, 576, 22, 22],[1, 1152, 11, 11]

        x1,x2,x3,x4=self.gam1(x1),self.gam2(x2),self.gam3(x3),self.gam4(x4)

        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)

        x = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        x = self.up3(x, x1)
        # out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')


        o = self.head(x)  # segmentation logits
        s = self.field_head(x)  # >>> 新增：预测的形状场
        with torch.no_grad():  # ★ CCS 不反传梯度
            out_ccs = self.ccs_sigmoid(o, s)

        # out = self.ccs_sigmoid(o, s)  # >>> 新增：通过CCS-Sigmoid生成最终mask
        # out = F.interpolate(out, scale_factor=4, mode='bilinear')

        out_ccs = F.interpolate(out_ccs, scale_factor=4, mode='bilinear')
        o = F.interpolate(o, scale_factor=4, mode='bilinear')

        return o, out_ccs, out1, out2

        # return out, out1, out2

if __name__ == "__main__":
    with torch.no_grad():
        model = TSAMNet().cuda()
        x = torch.randn(1, 3, 352, 352).cuda()
        out, out1, out2 = model(x)
        #print(out.shape, out1.shape, out2.shape)
        print(TSAMNet())

