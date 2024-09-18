from PIL import Image
from sklearn.cluster import KMeans
from cuml.cluster import KMeans  # 使用 cuML 的 KMeans
import numpy as np
import cupy as cp  # 使用 cuPy 来处理 GPU 上的数组

# K-Means 颜色量化（使用 cuML 加速）
# 完全在 GPU 上进行 K-Means 颜色量化
def kmeans_quantization(image, n_colors):
    # 将图像数据转换为 RGB 并转为 cuPy 数组
    img = image.convert("RGB")
    img_np = cp.array(np.array(img))  # 直接将 NumPy 数组转换为 cuPy 数组
    w, h, d = img_np.shape

    # 将图像展平为二维数组 (像素数, 颜色通道)
    img_flat = img_np.reshape((w * h, d))

    # 使用 cuML 的 KMeans 进行颜色聚类
    kmeans = KMeans(n_clusters=n_colors, random_state=0,max_iter=100)
    kmeans.fit(img_flat)

    # 获取聚类的质心和标签（这些仍然是 cuPy 数组）
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    # 使用质心重建量化后的图像
    quantized_img_flat = centroids[labels]

    # 将量化后的图像重构为原始形状 (w, h, d)
    quantized_img = quantized_img_flat.reshape((w, h, d))
    # 将 CuPy 数组从 GPU 转换为 NumPy 数组
    quantized_img_cpu = cp.asnumpy(quantized_img)

    # 将 NumPy 数组转换为 uint8 类型
    quantized_img_uint8 = np.uint8(quantized_img_cpu)

    # 使用 PIL 将 NumPy 数组转换为图像
    return Image.fromarray(quantized_img_uint8)


# Median Cut 颜色量化
def median_cut_quantization(image, n_colors):
    quantized_img = image.convert("P", palette=Image.ADAPTIVE, colors=n_colors)
    return quantized_img.convert("RGB")


# Floyd-Steinberg Dithering
def floyd_steinberg_dithering(image, n_colors):
    # 将图像转换为 RGB，并转为 numpy 数组
    img = image.convert("RGB")
    img_np = np.array(img, dtype=float)  # 使用浮点数进行计算

    # 获取图像的宽度和高度
    h, w, _ = img_np.shape

    # 生成调色板，用于图像的颜色量化
    palette = np.linspace(0, 255, n_colors, dtype=int)

    # 遍历图像像素并应用 Floyd-Steinberg Dithering
    for y in range(h):
        for x in range(w):
            original_pixel = img_np[y, x]
            # 找到该像素的最近调色板颜色
            new_pixel = np.array([min(palette, key=lambda p: abs(p - v)) for v in original_pixel])
            # 计算误差
            error = original_pixel - new_pixel
            # 替换当前像素
            img_np[y, x] = new_pixel

            # 将误差分散到邻近像素
            if x + 1 < w:
                img_np[y, x + 1] += error * 7 / 16
            if y + 1 < h:
                if x - 1 >= 0:
                    img_np[y + 1, x - 1] += error * 3 / 16
                img_np[y + 1, x] += error * 5 / 16
                if x + 1 < w:
                    img_np[y + 1, x + 1] += error * 1 / 16

    # 将图像转换回 uint8 并返回一个 PIL 图像
    return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))


# Median Cut with Perceptual Weighting 颜色量化

def median_cut_perceptual_weighting(image, n_colors):
    # 将图像转换为 RGB 并转换为 cuPy 数组
    img = image.convert("RGB")
    img_cp = cp.array(np.array(img))

    # 定义 RGB 通道的感知加权 (也使用 cuPy 数组)
    perceptual_weights = cp.array([0.299, 0.587, 0.114])

    # 计算加权后的颜色距离
    weighted_img_cp = img_cp.astype(float)
    for i in range(3):  # 对 R, G, B 通道分别加权
        weighted_img_cp[:, :, i] *= perceptual_weights[i]

    # 将 cuPy 数组转换回 NumPy 数组以便使用 Pillow 进行颜色量化
    weighted_image = Image.fromarray(cp.asnumpy(cp.uint8(weighted_img_cp)))  # 不对图像像素值进行改变
    quantized_img = weighted_image.convert("P", palette=Image.ADAPTIVE, colors=n_colors)

    return quantized_img.convert("RGB")