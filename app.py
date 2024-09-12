import gradio as gr
from PIL import Image
from color_quantization import (
    kmeans_quantization,
    median_cut_quantization,
    floyd_steinberg_dithering,
    median_cut_perceptual_weighting
)
from pixelation import pixelate_image


# 定义像素化和颜色量化处理函数
def pixelate_and_quantize(image, pixel_size, n_colors, method, interpolation):
    # 对图像进行像素化
    pixelated_img = pixelate_image(image, pixel_size, interpolation)

    # 根据用户选择的量化方法进行颜色量化
    if method == "K-Means":
        quantized_img = kmeans_quantization(pixelated_img, n_colors)
    elif method == "Median Cut":
        quantized_img = median_cut_quantization(pixelated_img, n_colors)
    elif method == "Floyd-Steinberg Dithering":
        quantized_img = floyd_steinberg_dithering(pixelated_img, n_colors)
    elif method == "Median Cut (Perceptual Weighting)":
        quantized_img = median_cut_perceptual_weighting(pixelated_img, n_colors)
    else:
        raise ValueError(f"未知的量化方法: {method}")

    return quantized_img


# 创建 Gradio 界面
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## 像素化图片工具 (邻近采样 + 颜色量化)")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="上传图片")

            with gr.Column():
                pixel_size_slider = gr.Slider(minimum=2, maximum=50, value=10, step=1, label="选择像素块大小")
                color_slider = gr.Slider(minimum=2, maximum=64, value=16, step=1, label="选择颜色数量")
                method_radio = gr.Radio(
                    choices=["K-Means", "Median Cut", "Floyd-Steinberg Dithering", "Median Cut (Perceptual Weighting)"],
                    value="K-Means",
                    label="选择颜色量化方法"
                )
                # 添加更多插值方法：Nearest, Bilinear, Bicubic, Lanczos
                interpolation_radio = gr.Radio(
                    choices=["Nearest", "Bilinear", "Bicubic", "Lanczos"],
                    value="Nearest",
                    label="选择插值方法"
                )

        image_output = gr.Image(label="像素化和量化后的图片")
        btn = gr.Button("生成像素化图片")

        btn.click(fn=pixelate_and_quantize, inputs=[image_input, pixel_size_slider, color_slider, method_radio, interpolation_radio], outputs=image_output)

    return demo


if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()