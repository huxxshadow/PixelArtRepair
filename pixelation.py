from PIL import Image

def pixelate_image(image, pixel_size, interpolation):
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