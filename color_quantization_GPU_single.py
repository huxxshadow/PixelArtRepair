from PIL import Image
from cuml.cluster import KMeans  # 使用 cuML 的 KMeans
import numpy as np
import cupy as cp  # 使用 cuPy 来处理 GPU 上的数组
global operating_image
import gol

# K-Means 颜色量化（使用 cuML 加速）
# 完全在 GPU 上进行 K-Means 颜色量化
def kmeans_quantization_get(image, n_colors):
    # 将图像数据转换为 RGB 并转为 cuPy 数组
    img = image.convert("RGB")
    img_np = cp.asarray(img)  # 直接将 NumPy 数组转换为 cuPy 数组
    w, h, d = img_np.shape

    # 将图像展平为二维数组 (像素数, 颜色通道)
    img_flat = img_np.reshape((w * h, d))

    # 使用 cuML 的 KMeans 进行颜色聚类
    kmeans = KMeans(n_clusters=n_colors, random_state=0, max_iter=50,init="k-means++")
    kmeans.fit(img_flat)

    # 获取聚类的质心和标签（这些仍然是 cuPy 数组）
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    gol.set_value("labels", labels)

    return cp.asnumpy(centroids)


def kmeans_quantization_rebuild(centroids):
    image = gol.get_value("pixelated_img")
    # 将图像数据转换为 RGB 并转为 cuPy 数组
    img = image.convert("RGB")
    img_np = cp.asarray(img)  # 直接将 NumPy 数组转换为 cuPy 数组
    w, h, d = img_np.shape
    labels = gol.get_value("labels")

    centroids=cp.asarray(centroids)
    # 使用质心重建量化后的图像
    quantized_img_flat = centroids[labels]

    # 将量化后的图像重构为原始形状 (w, h, d)
    quantized_img = quantized_img_flat.reshape((w, h, d))

    # 将 CuPy 数组从 GPU 转换为 NumPy 数组
    quantized_img_cpu = cp.asnumpy(quantized_img)

    # 将 NumPy 数组转换为 uint8 类型
    quantized_img_uint8 = np.uint8(quantized_img_cpu)

    gol.set_value('img', quantized_img_uint8)


    return quantized_img_uint8
    # 使用 PIL 将 NumPy 数组转换为图像
    # return Image.fromarray(quantized_img_uint8), cp.asnumpy(centroids)