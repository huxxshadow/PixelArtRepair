from PIL import Image
from sklearn.cluster import KMeans
import numpy as np


# K-Means 颜色量化
def kmeans_quantization(image, n_colors):
    # 将图像转换为 RGB 模式
    img = image.convert("RGB")

    # 将图像数据转换为 NumPy 数组
    img_np = np.array(img)
    w, h, d = img_np.shape

    # 将像素数据展平为 (w*h, 3) 的二维数组
    img_flat = img_np.reshape((w * h, d))

    # 使用 KMeans 聚类，n_clusters = n_colors，表示调色板的颜色数量
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(img_flat)

    # 获取聚类的中心颜色
    centroids = kmeans.cluster_centers_.astype(int)

    # 将每个像素映射到最近的聚类中心
    labels = kmeans.labels_
    quantized_img_flat = centroids[labels]

    # 将量化后的像素重新恢复为原图形状
    quantized_img = quantized_img_flat.reshape((w, h, d))

    # 将 NumPy 数组转换回 PIL 图像
    quantized_img_pil = Image.fromarray(np.uint8(quantized_img))

    return quantized_img_pil


# Median Cut 颜色量化
def median_cut_quantization(image, n_colors):
    # 使用 Pillow 提供的 Median Cut 量化方法
    quantized_img = image.convert("P", palette=Image.MEDIANCUT, colors=n_colors)
    return quantized_img


# Octree 颜色量化
def octree_quantization(image, n_colors):
    # 使用 Pillow 提供的 Octree 量化方法
    quantized_img = image.convert("P", palette=Image.FASTOCTREE, colors=n_colors)
    return quantized_img