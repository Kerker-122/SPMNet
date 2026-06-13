import torch
import time
import numpy as np

# ================= 配置区域 =================
# 1. 导入你的模型
# from networks import MySegmentationModel as CurrentModel
from TSAMNet import TSAMNet as CurrentModel

# 2. 设置参数
INPUT_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ===========================================

try:
    from thop import profile
except ImportError:
    print("错误: 请先安装 thop 库 -> pip install thop")
    exit()


def cal_params_flops(model, input_size, device):
    """
    计算参数量 (Params) 和 计算量 (GFLOPs)
    """
    # 构造一个 dummy input
    input_tensor = torch.randn(1, 3, input_size, input_size).to(device)

    # 切换到 eval 模式
    model.eval()

    # 使用 thop 计算
    # 注意：thop 可能会打印一些日志，可以忽略
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)

    return params, flops


def cal_fps(model, input_size, device, iterations=100):
    """
    计算 FPS (Frames Per Second)
    """
    model.eval()
    input_tensor = torch.randn(1, 3, input_size, input_size).to(device)

    # 1. Warm-up (预热)
    # GPU 刚开始运行时需要初始化缓存，预热可以消除这部分时间偏差
    print("Warming up...")
    with torch.no_grad():
        for _ in range(20):
            _ = model(input_tensor)

    # 2. 正式计时
    print(f"Starting FPS test ({iterations} iterations)...")

    # 必须使用 torch.cuda.Event 进行准确的 GPU 计时
    if device.type == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        # 等待前面的操作完成
        torch.cuda.synchronize()

        starter.record()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(input_tensor)
        ender.record()

        # 等待所有操作完成
        torch.cuda.synchronize()

        # 计算时间 (ms -> s)
        curr_time = starter.elapsed_time(ender) / 1000.0
    else:
        # CPU 计时
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(input_tensor)
        curr_time = time.time() - start_time

    fps = iterations / curr_time
    return fps


def main():
    print(f"Device: {DEVICE}")
    print(f"Input Size: {INPUT_SIZE}x{INPUT_SIZE}")

    # --- 第 1 部分：计算 Params 和 GFLOPs ---
    model = CurrentModel().to(DEVICE)  # 第一次实例化

    print("-" * 30)
    params, flops = cal_params_flops(model, INPUT_SIZE, DEVICE)
    print(f"Parameters (M) : {params / 1e6 :.4f} M")
    print(f"GFLOPs         : {flops / 1e9 :.4f} G")
    print("-" * 30)

    # --- 关键修改：销毁旧模型，重新实例化新模型 ---
    del model  # (可选) 删除旧模型释放显存
    torch.cuda.empty_cache()  # (可选) 清理显存

    print("Re-instantiating model for FPS test...")
    model = CurrentModel().to(DEVICE)  # <--- 第二次实例化，得到一个干净的模型

    # --- 第 2 部分：计算 FPS ---
    fps = cal_fps(model, INPUT_SIZE, DEVICE, iterations=200)
    print(f"FPS            : {fps:.2f}")
    print("-" * 30)


if __name__ == '__main__':
    main()