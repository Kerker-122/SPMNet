import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from TSAMNet import TSAMNet

class FeatureMapVisualizer:
    def __init__(self, save_dir="ablation_vis_separate"):
        self.hooks = []
        self.features = {}
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def hook_fn(self, module, input, output, name):
        """
        钩子函数：拦截模块的输入和输出
        """
        # input 是一个 tuple，input[0] 通常是我们需要的张量

        # === 特殊处理 CCS 模块 ===
        # CCS 的输入是 (logits, shape_field)，我们需要分别保存
        if name == "CCSSigmoid":
            self.features[name] = {
                "input_logits": input[0].detach(),  # o
                "input_shape": input[1].detach(),  # s
                "output_final": output.detach()  # u (经过CCS处理后的结果)
            }
        else:
            # === 通用模块 (CFEM, SAG) ===
            # input[0] 是进入模块前的特征 (Before)
            # output 是出模块后的特征 (After)
            self.features[name] = {
                "before": input[0].detach(),
                "after": output.detach()
            }

    def register_hooks(self, model):
        """
        注册三个指定位置的 Hook
        """
        # 1. CFEM (对应代码中的 TwinAttn)
        # 选择 gam3 (576通道) 或 gam4 (1152通道) 可以看到更深层的语义变化
        if hasattr(model, 'gam3'):
            model.gam3.register_forward_hook(
                lambda m, i, o: self.hook_fn(m, i, o, "CFEM")
            )

        # 2. SAG (对应 Up 模块中的 sag)
        # 选择 up3 (最后一次上采样融合)，最接近最终输出，效果最直观
        if hasattr(model, 'up3'):
            model.up3.sag.register_forward_hook(
                lambda m, i, o: self.hook_fn(m, i, o, "SAG")
            )

        # 3. CCS (对应 ccs_sigmoid)
        if hasattr(model, 'ccs_sigmoid'):
            model.ccs_sigmoid.register_forward_hook(
                lambda m, i, o: self.hook_fn(m, i, o, "CCSSigmoid")
            )

    def process_tensor(self, tensor):
        """
        将 Tensor 转换为热力图数据 (numpy array)
        """
        # 如果是 [B, N, C] 格式 (SAM2 Token)，尝试 reshape
        if tensor.dim() == 3:
            b, n, c = tensor.shape
            h = w = int(n ** 0.5)
            tensor = tensor.transpose(1, 2).view(b, c, h, w)

        # 对通道求平均 [B, C, H, W] -> [B, H, W]
        heatmap = torch.mean(tensor, dim=1)

        # 归一化到 0-1
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = heatmap.cpu().numpy()[0]  # 取 batch 中第一张图

        return heatmap

    def save_heatmap(self, heatmap_data, filename):
        """
        绘制并单独保存一张热力图
        """
        plt.figure(figsize=(4, 4))  # 画布大小
        plt.imshow(heatmap_data, cmap='jet')  # 使用 jet 伪彩色
        plt.axis('off')  # 关闭坐标轴

        # 去除白边保存
        full_path = os.path.join(self.save_dir, filename)
        plt.savefig(full_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        print(f"Saved: {full_path}")

    def save_all_images(self, img_prefix="vis"):
        """
        遍历所有抓取的特征并保存
        """
        for module_name, data in self.features.items():

            # === 处理 CCS (3张图) ===
            if module_name == "CCSSigmoid":
                # 1. Input Logits
                hm_logits = self.process_tensor(data["input_logits"])
                self.save_heatmap(hm_logits, f"{img_prefix}_{module_name}_1_Input_Logits.png")

                # 2. Input Shape
                hm_shape = self.process_tensor(data["input_shape"])
                self.save_heatmap(hm_shape, f"{img_prefix}_{module_name}_2_Input_Shape.png")

                # 3. Output
                hm_out = self.process_tensor(data["output_final"])
                self.save_heatmap(hm_out, f"{img_prefix}_{module_name}_3_Output.png")

            # === 处理 CFEM 和 SAG (2张图: Before/After) ===
            else:
                # 1. Before
                hm_before = self.process_tensor(data["before"])
                self.save_heatmap(hm_before, f"{img_prefix}_{module_name}_1_Before.png")

                # 2. After
                hm_after = self.process_tensor(data["after"])
                self.save_heatmap(hm_after, f"{img_prefix}_{module_name}_2_After.png")


# ================= 使用示例 =================
if __name__ == "__main__":
    # 1. 初始化模型 (请使用之前教你的正确加载权重的方法)
    print("Initializing Model...")
    model = TSAMNet(checkpoint_path=None).cuda()
    state_dict = torch.load("./checkpoints/SAM2-1/TSAMNet60.pth", map_location='cpu')
    if 'model' in state_dict: state_dict = state_dict['model']
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 2. 注册可视化工具
    visualizer = FeatureMapVisualizer(save_dir="ablation_result_images")
    visualizer.register_hooks(model)

    # 3. 准备输入图片
    img_path = "./compare/img/7.jpg"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (352, 352))
    input_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).cuda()

    # 4. 推理
    with torch.no_grad():
        model(input_tensor)

    # 5. 生成并保存单独的图片
    visualizer.save_all_images(img_prefix="Sample1")