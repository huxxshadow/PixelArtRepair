import cv2
import numpy as np
from collections import Counter

def detect_pixel_block_size(image_path):
    # 读取图像
    img = cv2.imread(image_path)

    # 转为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 获取图像的高和宽
    height, width = gray.shape

    # 创建一个存储颜色变化的矩阵
    diff_matrix = np.zeros((height, width), dtype=np.uint8)

    # 遍历图像，检测颜色变化
    for i in range(1, height):
        for j in range(1, width):
            if gray[i, j] != gray[i-1, j] or gray[i, j] != gray[i, j-1]:
                diff_matrix[i, j] = 255  # 标记颜色变化的地方

    # 基于差异矩阵，使用轮廓检测寻找颜色块的边框
    contours, _ = cv2.findContours(diff_matrix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算每个轮廓的大小
    block_sizes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        block_sizes.append((w, h))

    # 统计最常见的块尺寸
    size_counter = Counter(block_sizes)
    most_common_size = size_counter.most_common(1)[0][0]

    return most_common_size

# 示例使用
image_path = 'NewMoon.png'  # 替换为你的图片路径
block_size = detect_pixel_block_size(image_path)
print(f"检测到的像素块尺寸为: {block_size}")