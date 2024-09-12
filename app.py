import gradio as gr
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np


# 定义 K-Means 颜色量化函数
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


# 定义 Median Cut 颜色量化函数（使用 Pillow 的 ADAPTIVE 调色板模式）
def median_cut_quantization(image, n_colors):
    """
    使用 Pillow 的 ADAPTIVE 调色板模式进行 Median Cut 颜色量化。

    参数:
    - image: 输入的 PIL 图像对象
    - n_colors: 颜色数量

    返回:
    - 量化后的 PIL 图像对象
    """
    # 使用 Pillow 的 ADAPTIVE 调色板模式进行颜色量化
    quantized_img = image.convert("P", palette=Image.ADAPTIVE, colors=n_colors)

    # 将量化后的图像从调色板模式转换回 RGB 模式以便显示
    return quantized_img.convert("RGB")


# 定义像素化和颜色量化处理函数
def pixelate_and_quantize(image, pixel_size, n_colors, method):
    # 将输入图像转为 RGB 模式
    img = image.convert("RGB")

    # 获取原图像的尺寸
    width, height = img.size

    # 将图像缩小到原始尺寸的 1/pixel_size
    small_img = img.resize(
        (width // pixel_size, height // pixel_size),
        resample=Image.NEAREST
    )

    # 再将图像放大到原始尺寸
    pixelated_img = small_img.resize(
        (width, height),
        resample=Image.NEAREST
    )

    # 根据用户选择的量化方法进行颜色量化
    if method == "K-Means":
        quantized_img = kmeans_quantization(pixelated_img, n_colors)
    elif method == "Median Cut":
        quantized_img = median_cut_quantization(pixelated_img, n_colors)
    else:
        raise ValueError(f"未知的量化方法: {method}")

    return quantized_img


# 创建 Gradio 界面
def gradio_interface():
    # 创建 Gradio 输入部分，允许上传图片与选择像素大小、颜色数量和量化方法
    with gr.Blocks() as demo:
        gr.Markdown("## 像素化图片工具 (邻近采样 + 颜色量化)")

        # 输入图片组件
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="上传图片")

            with gr.Column():
                pixel_size_slider = gr.Slider(
                    minimum=2,
                    maximum=50,
                    value=10,
                    step=1,
                    label="选择像素块大小"
                )
                color_slider = gr.Slider(
                    minimum=2,
                    maximum=64,
                    value=16,
                    step=1,
                    label="选择颜色数量"
                )
                method_radio = gr.Radio(
                    choices=["K-Means", "Median Cut"],
                    value="K-Means",
                    label="选择颜色量化方法"
                )

        # 输出像素化后的图像
        image_output = gr.Image(label="像素化和量化后的图片")

        # 添加按钮进行处理
        btn = gr.Button("生成像素化图片")

        # 定义按钮点击触发的事件
        btn.click(
            fn=pixelate_and_quantize,
            inputs=[image_input, pixel_size_slider, color_slider, method_radio],
            outputs=image_output
        )

    return demo


# 启动 Gradio 界面
if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()