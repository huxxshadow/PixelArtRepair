import gradio as gr
from PIL import Image
from color_quantization import (
    kmeans_quantization,
    median_cut_quantization,
    floyd_steinberg_dithering,
    median_cut_perceptual_weighting
)
from pixelation import pixelate_image, mosaic_pixelation, oil_paint_pixelation, hierarchical_pixelation
import numpy as np
from collections import Counter
import math


# 定义像素化和颜色量化处理函数
def pixelate_and_quantize(image, pixel_size, n_colors, methods, interpolation, pixelation_types):
    results = []

    # 根据用户选择的像素化方法进行像素化处理
    for pixelation_type in pixelation_types:
        if pixelation_type == "Classic Nearest Neighbor":
            pixelated_img = pixelate_image(image, pixel_size, interpolation)
        elif pixelation_type == "Mosaic":
            pixelated_img = mosaic_pixelation(image, pixel_size)
        elif pixelation_type == "Oil Painting":
            pixelated_img = oil_paint_pixelation(image, pixel_size)
        elif pixelation_type == "Hierarchical":
            pixelated_img = hierarchical_pixelation(image, pixel_size, pixel_size * 2)
        else:
            raise ValueError(f"未知的像素化方法: {pixelation_type}")

        # 根据用户选择的量化方法进行颜色量化，并将量化后的图像添加到列表中
        for method in methods:
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

            # 只返回量化后的图像
            results.append(quantized_img)

    return results


# Gradio 用户界面
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## 像素化图片工具 (邻近采样 + 颜色量化)")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="上传图片")

            with gr.Column():
                pixel_size_slider = gr.Slider(minimum=2, maximum=50, value=10, step=1, label="选择像素块大小")
                color_slider = gr.Slider(minimum=2, maximum=64, value=16, step=1, label="选择颜色数量")

                method_checkboxes = gr.CheckboxGroup(
                    choices=["K-Means", "Median Cut", "Floyd-Steinberg Dithering", "Median Cut (Perceptual Weighting)"],
                    value=["K-Means"],
                    label="选择颜色量化方法"
                )
                interpolation_radio = gr.Radio(
                    choices=["Nearest", "Bilinear", "Bicubic", "Lanczos"],
                    value="Nearest",
                    label="选择插值方法"
                )
                pixelation_checkboxes = gr.CheckboxGroup(
                    choices=["Classic Nearest Neighbor", "Mosaic", "Oil Painting", "Hierarchical"],
                    value=["Classic Nearest Neighbor"],
                    label="选择像素化方法"
                )

        btn = gr.Button("生成像素化图片")

        @gr.render(inputs=[image_input,pixel_size_slider,color_slider, method_checkboxes, interpolation_radio,pixelation_checkboxes], triggers=[btn.click])
        def show_pictures(img_input,pixel_size,n_colors,methods, interpolation, pixelation_types):
            num_outputs = len(methods) * len(pixelation_types)
            num_outputs = min(num_outputs, 16)
            images = pixelate_and_quantize(img_input, pixel_size, n_colors, methods, interpolation, pixelation_types)
            cols = math.ceil(math.sqrt(num_outputs))
            rows = math.ceil(num_outputs / cols)
            for i in range(rows):
                single_row= gr.Row()
                with single_row:
                    for j in range(cols):
                        single_col = gr.Column()
                        idx = i * cols + j
                        with single_col:
                            if idx < num_outputs:
                                gr.Image(images[idx],label=f"像素化和量化后的图片 {idx + 1}")


    return demo


# 启动 Gradio 界面
if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()