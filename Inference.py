import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import time

# ================= 配置区域 =================
# 1. 导入你的模型
from TSAMNet import TSAMNet as CurrentModel


# 2. 设置路径
INPUT_FOLDER = './compare/img'  # 存放 1.jpg, 2.jpg... 的文件夹
OUTPUT_FOLDER = ('./compare/TSAMNet50')  # 结果保存文件夹
WEIGHTS_PATH = './checkpoints/SAM2-1/TSAMNet50.pth'

# 3. 参数设置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_SIZE = 352
THRESHOLD = 0.5  # 二值化阈值


def load_model(model_class, weights_path):
    print(f"Loading model architecture...")
    model = model_class().to(DEVICE)

    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location=DEVICE)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        print(f"Warning: Weights file not found at {weights_path}, using random init.")

    model.eval()
    return model


def preprocess_image(image_path, input_size):
    """读取并预处理单张图片"""
    try:
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0)
        return img_tensor.to(DEVICE)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def postprocess_and_save(output_tensor, save_path, threshold=0.5):
    """二值化处理并保存"""
    # 1. Sigmoid (如果模型输出是Logits)
    # 如果你的 forward 已经包含 sigmoid，请注释掉下面这行
    res = torch.sigmoid(output_tensor)

    # 2. 移除维度 -> [H, W]
    res = res.squeeze().data.cpu().numpy()

    # 3. 二值化 (关键步骤)
    res = (res > threshold).astype(np.uint8) * 255

    # 4. 保存
    mask = Image.fromarray(res)
    mask.save(save_path)


def main():
    # 1. 准备环境
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output directory: {OUTPUT_FOLDER}")

    # 2. 加载模型 (只加载一次)
    model = load_model(CurrentModel, WEIGHTS_PATH)

    # 3. 获取文件列表
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_extensions)]
    # 按文件名数字排序 (可选，方便观察进度: 1.jpg, 2.jpg, 10.jpg...)
    try:
        image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        image_files.sort()  # 如果文件名不是纯数字，则按字母排序

    print(f"Found {len(image_files)} images. Starting inference...")

    start_time = time.time()

    # 4. 循环处理
    with torch.no_grad():
        for i, img_name in enumerate(image_files):
            img_path = os.path.join(INPUT_FOLDER, img_name)

            # 构造保存路径： 1.jpg -> 1.png
            save_name = os.path.splitext(img_name)[0] + ".png"
            save_path = os.path.join(OUTPUT_FOLDER, save_name)

            # 预处理
            input_tensor = preprocess_image(img_path, INPUT_SIZE)
            if input_tensor is None: continue

            # 推理
            output = model(input_tensor)

            # 处理深监督输出 (取第一个)
            if isinstance(output, (list, tuple)):
                output = output[0]

            # 后处理并保存
            postprocess_and_save(output, save_path, THRESHOLD)

            # 打印进度 (每10张或最后一张打印一次)
            if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
                print(f"Processed {i + 1}/{len(image_files)}: {save_name}")

    total_time = time.time() - start_time
    print(f"Done! All results saved to {OUTPUT_FOLDER}")
    print(f"Total time: {total_time:.2f}s, Average FPS: {len(image_files) / total_time:.2f}")


if __name__ == '__main__':
    main()