import os

# 设置路径
image_dir = './MAS3K/test/img'  # 修改成你的真实路径
output_file = './MAS3K/test/test.txt'

# 获取所有图像文件名（去掉扩展名）
with open(output_file, 'w') as f:
    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0]  # 去掉扩展名
            f.write(name + '\n')

print(f"✅ 已生成 {output_file}，共 {len(os.listdir(image_dir))} 个文件")
