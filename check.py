import os

# 把这里的路径换成你报错的那个 .pth 文件的绝对路径
file_path = r"D:\Code\SAM2-UNet-main\SAM2-UNet-main\checkpoints\SAM2-1\AttentionUNet.pth"


def main() :
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / 1024 / 1024
        print(f"当前文件大小: {size_mb:.2f} MB")

        # 读取文件前50个字符看看是不是文本
        try:
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read(50)
                print(f"文件头预览: {content}")
                if "git-lfs" in content:
                    print("【诊断结果】: 这是一个 Git LFS 指针文件，不是权重！请去网页端手动下载原文件。")
                elif size_mb < 1:
                    print("【诊断结果】: 文件太小，可能是下载中断或空文件。")
                else:
                    print("【诊断结果】: 大小看起来正常，但内部结构损坏，建议重新下载。")
        except Exception as e:
            print(f"读取出错: {e}")
    else:
        print("找不到文件路径，请检查路径是否正确。")



if __name__ == '__main__':
    main()