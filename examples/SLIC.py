# 这是一个使用 Python 中 scikit-image 库调用 SLIC 算法的简短示例。
# 在运行前，请确保已安装所需库：pip install scikit-image matplotlib numpy

import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from skimage.data import astronaut # 使用内置的'宇航员'图像作为示例数据

# 1. 准备图像数据
image = astronaut()

# 2. 定义 SLIC 参数
# n_segments: 期望生成的超像素数量
# compactness: 颜色相似度与空间邻近度的权衡因子 (值越高，超像素越规则)
N_SEGMENTS = 250
COMPACTNESS = 10

# 3. 核心库调用：运行 SLIC 算法
# SLIC 返回一个标签数组，其中每个唯一的标签对应一个超像素区域。
print(f"--- 正在执行 SLIC 算法 ({N_SEGMENTS} 个超像素) ---")
segments = slic(
    image,
    n_segments=N_SEGMENTS,
    compactness=COMPACTNESS,
    sigma=1.0, # 图像平滑因子
    start_label=1
)
print("SLIC 执行完毕。")
print(f"生成的唯一超像素数量: {len(np.unique(segments))}")

# 4. 可视化结果
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# mark_boundaries 函数用于在原始图像上标记超像素的边界
ax.imshow(mark_boundaries(image, segments))
ax.set_title(f"SLIC Superpixels (Segments: {len(np.unique(segments))})")
ax.axis('off')
plt.show()