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
    img = image.convert("RGB")
    width, height = img.size

    # 使用比例调整 pixel_size，但确保至少为 1
    # 基准值 512 可根据需要调整
    scale_factor = max(1, min(width, height) // 512)
    adjusted_pixel_size = max(1, pixel_size * scale_factor)

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

    # 确保输出尺寸至少为1x1
    small_width = max(1, width // adjusted_pixel_size)
    small_height = max(1, height // adjusted_pixel_size)

    small_img = img.resize(
        (small_width, small_height),
        resample=resample_method
    )

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

    # 使用比例调整 pixel_size，但确保至少为 1
    scale_factor = max(1, min(w, h) // 512)  # 根据需要调整基准值
    adjusted_pixel_size = max(1, pixel_size * scale_factor)

    for y in range(0, h, adjusted_pixel_size):
        for x in range(0, w, adjusted_pixel_size):
            block = img_np[y:y + adjusted_pixel_size, x:x + adjusted_pixel_size]
            mean_color = block.mean(axis=(0, 1)).astype(int)
            img_np[y:y + adjusted_pixel_size, x:x + adjusted_pixel_size] = mean_color

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

    # 使用比例调整 pixel_size，但确保至少为 1
    scale_factor = max(1, min(w, h) // 512)  # 根据需要调整基准值
    adjusted_pixel_size = max(1, pixel_size * scale_factor)

    for y in range(0, h, adjusted_pixel_size):
        for x in range(0, w, adjusted_pixel_size):
            block = img_np[y:y + adjusted_pixel_size, x:x + adjusted_pixel_size]
            block_colors = [tuple(color) for color in block.reshape(-1, 3)]
            most_common_color = Counter(block_colors).most_common(1)[0][0]
            img_np[y:y + adjusted_pixel_size, x:x + adjusted_pixel_size] = most_common_color

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

    # 使用比例调整 pixel_size，但确保至少为 1
    scale_factor = max(1, min(w, h) // 512)  # 根据需要调整基准值
    adjusted_min_pixel_size = max(1, min_pixel_size * scale_factor)
    adjusted_max_pixel_size = max(1, max_pixel_size * scale_factor)

    # 防止步长为0
    step = max((adjusted_max_pixel_size - adjusted_min_pixel_size) // max(w // adjusted_min_pixel_size, 1), 1)

    for pixel_size in range(adjusted_min_pixel_size, adjusted_max_pixel_size + 1, step):
        for y in range(0, h, pixel_size):
            for x in range(0, w, pixel_size):
                block = img_np[y:y + pixel_size, x:x + pixel_size]
                mean_color = block.mean(axis=(0, 1)).astype(int)
                img_np[y:y + pixel_size, x:x + pixel_size] = mean_color

    return Image.fromarray(img_np)

