from PIL import Image
import numpy as np
from collections import Counter

def pixelate_image(image, pixel_size, interpolation="Nearest"):
    """
    对图像进行像素化。

    参数:
    - image: 输入的 PIL 图像对象
    - pixel_size: 像素块大小
    - interpolation: 插值方法 ("Nearest", "Bilinear", "Bicubic", "Lanczos")

    返回:
    - 像素化后的 PIL 图像对象
    """
    # 将输入图像转为 RGB 模式
    img = image.convert("RGB")

    # 获取原图像的尺寸
    width, height = img.size
    pixel_size = round(min(width,height)/1024)*pixel_size

    # 选择插值方式
    if interpolation == "Nearest":
        resample_method = Image.NEAREST
    elif interpolation == "Bilinear":
        resample_method = Image.BILINEAR
    elif interpolation == "Bicubic":
        resample_method = Image.BICUBIC
    elif interpolation == "Lanczos":
        resample_method = Image.LANCZOS
    else:
        raise ValueError(f"未知的插值方法: {interpolation}")

    # 第一步：缩小图像，使用邻近插值保持像素块的正方形效果
    small_img = img.resize(
        (width // pixel_size, height // pixel_size),
        resample=resample_method
    )

    # 第二步：放大图像，使用用户选择的插值方法
    pixelated_img = small_img.resize(
        (width, height),
        resample=resample_method
    )

    return pixelated_img


def mosaic_pixelation(image, pixel_size):
    """
    使用马赛克方法对图像进行像素化。

    参数:
    - image: 输入的 PIL 图像对象
    - pixel_size: 像素块大小

    返回:
    - 马赛克效果的 PIL 图像对象
    """
    img = image.convert("RGB")
    img_np = np.array(img)
    h, w, _ = img_np.shape
    pixel_size = round(min(w, h) / 1024) * pixel_size

    for y in range(0, h, pixel_size):
        for x in range(0, w, pixel_size):
            block = img_np[y:y + pixel_size, x:x + pixel_size]
            mean_color = block.mean(axis=(0, 1)).astype(int)
            img_np[y:y + pixel_size, x:x + pixel_size] = mean_color

    return Image.fromarray(img_np)


def oil_paint_pixelation(image, pixel_size):
    """
    使用油画滤镜方法对图像进行像素化。

    参数:
    - image: 输入的 PIL 图像对象
    - pixel_size: 像素块大小

    返回:
    - 油画滤镜效果的 PIL 图像对象
    """
    img = image.convert("RGB")
    img_np = np.array(img)
    h, w, _ = img_np.shape
    pixel_size = round(min(w, h) / 1024) * pixel_size
    for y in range(0, h, pixel_size):
        for x in range(0, w, pixel_size):
            block = img_np[y:y + pixel_size, x:x + pixel_size]
            block_colors = [tuple(color) for color in block.reshape(-1, 3)]
            most_common_color = Counter(block_colors).most_common(1)[0][0]
            img_np[y:y + pixel_size, x:x + pixel_size] = most_common_color

    return Image.fromarray(img_np)


def hierarchical_pixelation(image, min_pixel_size, max_pixel_size):
    """
    使用层次像素化方法对图像进行像素化。

    参数:
    - image: 输入的 PIL 图像对象
    - min_pixel_size: 最小像素块大小
    - max_pixel_size: 最大像素块大小

    返回:
    - 层次像素化效果的 PIL 图像对象
    """
    img = image.convert("RGB")
    img_np = np.array(img)
    h, w, _ = img_np.shape
    min_pixel_size = round(min(w, h) / 1024) * min_pixel_size
    max_pixel_size = round(min(w, h) / 1024) * max_pixel_size

    step = max((max_pixel_size - min_pixel_size) // (w // min_pixel_size), 1)

    for pixel_size in range(min_pixel_size, max_pixel_size + 1, step):
        for y in range(0, h, pixel_size):
            for x in range(0, w, pixel_size):
                block = img_np[y:y + pixel_size, x:x + pixel_size]
                mean_color = block.mean(axis=(0, 1)).astype(int)
                img_np[y:y + pixel_size, x:x + pixel_size] = mean_color

    return Image.fromarray(img_np)