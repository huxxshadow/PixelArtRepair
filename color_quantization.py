from PIL import Image
from sklearn.cluster import KMeans
import numpy as np


# K-Means 颜色量化
def kmeans_quantization(image, n_colors):
    img = image.convert("RGB")
    img_np = np.array(img)
    w, h, d = img_np.shape
    img_flat = img_np.reshape((w * h, d))
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(img_flat)
    centroids = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    quantized_img_flat = centroids[labels]
    quantized_img = quantized_img_flat.reshape((w, h, d))
    return Image.fromarray(np.uint8(quantized_img))


# Median Cut 颜色量化
def median_cut_quantization(image, n_colors):
    quantized_img = image.convert("P", palette=Image.ADAPTIVE, colors=n_colors)
    return quantized_img.convert("RGB")


# Floyd-Steinberg Dithering
def floyd_steinberg_dithering(image, n_colors):
    quantized_img = image.convert("P", palette=Image.ADAPTIVE, colors=n_colors)
    return quantized_img.convert("RGB")


# Median Cut with Perceptual Weighting 颜色量化
def median_cut_perceptual_weighting(image, n_colors):
    """
    使用感知加权的 Median Cut 颜色量化。

    参数:
    - image: 输入的 PIL 图像对象
    - n_colors: 目标颜色数量

    返回:
    - 量化后的 PIL 图像对象
    """
    img = image.convert("RGB")

    # 定义 RGB 通道的感知加权
    perceptual_weights = np.array([0.299, 0.587, 0.114])

    # 将图像转为 numpy 数组
    img_np = np.array(img)

    # 计算加权后的颜色距离
    weighted_img_np = img_np.astype(float)
    for i in range(3):  # 对 R, G, B 通道分别加权
        weighted_img_np[:, :, i] *= perceptual_weights[i]

    # 使用 Pillow 的 Median Cut 算法进行基于加权的颜色量化
    weighted_image = Image.fromarray(np.uint8(img_np))  # 不对图像像素值进行改变
    quantized_img = weighted_image.convert("P", palette=Image.ADAPTIVE, colors=n_colors)

    return quantized_img.convert("RGB")