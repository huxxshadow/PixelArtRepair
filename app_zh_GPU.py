import gradio as gr
from color_quantization_GPU import (
    kmeans_quantization,
    median_cut_quantization,
    floyd_steinberg_dithering,
    median_cut_perceptual_weighting
)
from pixelation import pixelate_image, mosaic_pixelation, oil_paint_pixelation, hierarchical_pixelation
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
            elif method == "MedianCut(PerceptualWeighting)":
                quantized_img = median_cut_perceptual_weighting(pixelated_img, n_colors)
            else:
                raise ValueError(f"未知的量化方法: {method}")

            # 只返回量化后的图像
            results.append(quantized_img)

    return results


# Gradio 用户界面
def gradio_interface():
    with gr.Blocks(title="Image Pixelation Tool") as demo:
        gr.Markdown("## 像素化图片工具 (AI像素修补)")
        gr.Markdown("""
                此工具提供了多种图像像素化效果和颜色量化算法，包括经典的邻近像素化、马赛克效果、油画效果以及层次像素化。
                用户可以根据需要选择不同的像素化方法并对图像进行颜色量化处理。

                ### 许可协议（License）
                本工具基于 **MPL 2.0（Mozilla Public License 2.0）** 协议发布。该协议允许修改和分发源代码，且允许商业使用，前提是对源代码的修改部分仍需保持开源。

                - **允许**：修改、分发、用于商业用途。
                - **限制**：修改过的源代码必须继续保持开源。

                [查看详细的 MPL 2.0 协议](https://www.mozilla.org/en-US/MPL/2.0/)

                **注意：如需将本工具用于商业用途，请联系作者以获得进一步的授权。**

                联系方式：huxxshadowhzy@gmail.com

                版权所有 © 2024
                """)
        with gr.Row():
            gr.Image(value="Example.png",interactive=False)

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="上传图片")

            with gr.Column():
                pixel_size_slider = gr.Slider(minimum=2, maximum=100, value=10, step=1, label="选择像素块大小")
                color_slider = gr.Slider(minimum=2, maximum=128, value=16, step=1, label="选择颜色数量")

                method_checkboxes = gr.CheckboxGroup(
                    choices=["K-Means", "Median Cut", "Floyd-Steinberg Dithering", "MedianCut(PerceptualWeighting)"],
                    value=["K-Means"],
                    label="选择颜色量化方法"
                )
                pixelation_checkboxes = gr.CheckboxGroup(
                    choices=["Classic Nearest Neighbor", "Mosaic", "Oil Painting", "Hierarchical"],
                    value=["Classic Nearest Neighbor"],
                    label="选择像素化方法"
                )
                interpolation_radio = gr.Radio(
                    choices=["Nearest", "Bilinear", "Bicubic", "Lanczos"],
                    value="Nearest",
                    label="选择插值方法(仅对Classic Nearest Neighbour有效)"
                )

        btn = gr.Button("生成像素化图片")

        @gr.render(inputs=[image_input, pixel_size_slider, color_slider, method_checkboxes, interpolation_radio,
                           pixelation_checkboxes], triggers=[btn.click])
        def show_pictures(img_input, pixel_size, n_colors, methods, interpolation, pixelation_types):
            num_outputs = len(methods) * len(pixelation_types)
            num_outputs = min(num_outputs, 16)
            images = pixelate_and_quantize(img_input, pixel_size, n_colors, methods, interpolation, pixelation_types)
            cols = math.ceil(math.sqrt(num_outputs))
            rows = math.ceil(num_outputs / cols)
            for i in range(rows):
                single_row = gr.Row()
                with single_row:
                    for j in range(cols):
                        single_col = gr.Column()
                        idx = i * cols + j
                        with single_col:
                            if idx < num_outputs:
                                # 计算当前的 pixelation 和 quantization 方法
                                current_pixelation = pixelation_types[idx // len(methods)]
                                current_method = methods[idx % len(methods)]
                                # 更新图片的标签，包含像素化方法和颜色量化方法
                                label = f"像素化: {current_pixelation}, 颜色量化: {current_method}"
                                gr.Image(images[idx], label=label, format="png")
                                idx += 1
        return demo


# 启动 Gradio 界面
if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()
