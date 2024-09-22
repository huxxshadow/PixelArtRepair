import gradio as gr
from color_quantization_GPU_single import kmeans_quantization_get,kmeans_quantization_rebuild
from pixelation import pixelate_image
from PIL import Image
import numpy as np
import cupy as cp
import gol
gol._init()
# 定义像素化和颜色量化处理函数
# def pixelate_and_quantize(image, pixel_size, n_colors):
#     # 只保留 Classic Nearest Neighbor 像素化方法
#     pixelated_img = pixelate_image(image, pixel_size)
#
#     # 只保留 K-Means 颜色量化
#     quantized_img, centroids = kmeans_quantization(pixelated_img, n_colors)
#
#     return quantized_img, centroids


def apply_selected_colors(image, selected_colors, tolerance=3):
    # 将图像转换为NumPy数组，然后转为CuPy数组（GPU上进行计算）
    img_np = np.array(image, dtype=np.uint8)
    img_cp = cp.array(img_np)  # 转换为CuPy数组，以便在GPU上操作
    h, w, d = img_cp.shape

    # 将选中的颜色从字符串转换为整数元组 (RGB格式)
    new_colors = [tuple(map(int, color.strip('()').split(','))) for color in selected_colors]

    # 将图像展平为二维数组 (像素数量, 颜色通道)
    img_flat = img_cp.reshape((-1, d))

    # 转换为 CuPy 数组以便进行 GPU 加速的向量化操作
    new_colors_cp = cp.array(new_colors, dtype=cp.uint8)

    # 对所有颜色进行批量计算，避免循环
    diff = cp.abs(img_flat[:, None, :] - new_colors_cp[None, :, :])  # 扩展维度并计算每个像素与所有颜色的差异
    within_tolerance = cp.all(diff <= tolerance, axis=2)  # 检查是否在容差范围内

    # 通过最大值操作找到符合条件的像素
    mask = cp.any(within_tolerance, axis=1)  # 只要有一个符合条件的颜色，掩码为True

    # 将不符合条件的像素设置为白色 (255, 255, 255)
    img_flat[~mask] = [255, 255, 255]

    # 将CuPy数组转换回NumPy数组，并恢复原始形状
    modified_img_np = cp.asnumpy(img_flat.reshape((h, w, d)))

    # 将NumPy数组转换回PIL图像并返回
    modified_img = Image.fromarray(modified_img_np.astype('uint8'))

    return modified_img

def rgb_to_hex(r, g, b):
    # 使用 f-string 格式化确保每个值转为两位十六进制数
    return f'#{r:02x}{g:02x}{b:02x}'
def hex_to_rgb(hex_color):
    # 从十六进制字符串中提取 R, G, B 值
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:], 16)
    return r, g, b
def rgb_list_to_hex(rgb_list):
    return [rgb_to_hex(r, g, b) for r, g, b in rgb_list]

def hex_list_to_rgb(hex_list):
    return [hex_to_rgb(hex_color) for hex_color in hex_list]

import colorsys

def rgb_to_hsv(rgb_color):
    """将 RGB 颜色转换为 HSV 颜色"""
    r, g, b = rgb_color
    return colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

def sort_colors_by_hue(int_centroids):
    """
    按照色相值对 RGB 颜色进行排序，返回排序后的 RGB 颜色
    :param int_centroids: 包含 RGB 颜色元组的列表
    :return: 按色相排序后的 RGB 颜色元组列表
    """
    # 将 RGB 转换为 HSV，并按色相 (Hue) 值排序，但保持原始 RGB 返回
    sorted_colors = sorted(int_centroids, key=lambda rgb: rgb_to_hsv(rgb)[0])
    return sorted_colors

# Gradio 用户界面
def gradio_interface():
    with gr.Blocks(title="Image Pixelation Tool") as demo:
        gr.Markdown("## 像素化图片工具 (AI像素修补)")

        with gr.Row():
            gr.Image(value="Example.png", interactive=False)

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="上传图片")

            with gr.Column():
                pixel_size_slider = gr.Slider(minimum=2, maximum=100, value=10, step=1, label="选择像素块大小")
                color_slider = gr.Slider(minimum=2, maximum=128, value=16, step=1, label="选择颜色数量")

        btn1 = gr.Button("量化颜色")

        modified_img = gr.Image(visible=True,type="numpy")
        color_quantization_checkboxes=gr.CheckboxGroup(visible=True)

        @gr.render(inputs=[], triggers=[color_quantization_checkboxes.change])
        def on_Colors_change():
            colors_cen = gol.get_value("raw_centroids")
            colors_16 = rgb_list_to_hex(colors_cen)
            picker_list = []
            with gr.Row():
                for i in range(len(colors_16)):
                    picker=gr.ColorPicker(value=colors_16[i], label=f"颜色{i}", interactive=True)
                    picker_list.append(picker)
            gol.set_value("picker_list", picker_list)
            btn2.click(
                on_generate_img,
                inputs=gol.get_value("picker_list"),
                outputs=[modified_img, color_checkboxes])


        btn2 = gr.Button("生成像素化图片")

        color_checkboxes = gr.CheckboxGroup(visible=True)



        def on_generate_color(img_input, pixel_size, n_colors):
            pixelated_img = pixelate_image(img_input, pixel_size)
            gol.set_value("pixelated_img", pixelated_img)
            centroids = kmeans_quantization_get(pixelated_img, n_colors)
            int_centroids = [tuple(map(int, centroid)) for centroid in centroids]
            gol.set_value("raw_centroids", int_centroids)
            color_choices = [(str(i), str(color)) for i, color in enumerate(int_centroids)]

            return gr.CheckboxGroup(
                choices=color_choices,
                interactive=False,
                value=[str(color) for color in int_centroids]
            )

        btn1.click(
            on_generate_color,
            inputs=[image_input, pixel_size_slider, color_slider],
            outputs=[color_quantization_checkboxes]
        )









        def on_generate_img(*args):

            new_centroids=list(args)
            # 对输入图像进行像素化和量化
            centroids = hex_list_to_rgb(new_centroids)

            quantized_img = kmeans_quantization_rebuild(centroids)

            # 将颜色转换为整数元组，并存储在全局变量中
            int_centroids = [tuple(map(int, centroid)) for centroid in centroids]

            # 按色相值对 int_centroids 排序，并保持 RGB 表示
            sorted_int_centroids = sort_colors_by_hue(int_centroids)

            gol.set_value("colors", sorted_int_centroids)

            # 为 CheckboxGroup 创建选项
            color_choices = [(str(i), str(color)) for i, color in enumerate(sorted_int_centroids)]

            # 返回量化后的图像和交互式 CheckboxGroup
            return quantized_img, gr.CheckboxGroup(
                choices=color_choices,
                interactive=True,
                value=[str(color) for color in sorted_int_centroids]
            )

        # 点击按钮时调用 on_generate 函数
        # btn2.click(
        #     on_generate_img,
        #     inputs=gol.get_value("picker_list"),
        #     outputs=[modified_img, color_checkboxes]
        # )
        @gr.render(inputs=[], triggers=[color_checkboxes.change])
        def on_Colors_change():
            colors_cen= gol.get_value("colors")
            colors_16=rgb_list_to_hex(colors_cen)
            with gr.Row():
                for i in range(len(colors_16)):
                    gr.ColorPicker(value=colors_16[i],label=f"颜色{i}",interactive=False)

        modify_btn = gr.Button("应用选择的颜色")
        final_img = gr.Image(visible=True,format="png")




        def on_apply_colors(selected_colors):
            img = apply_selected_colors(gol.get_value("img"), selected_colors)
            return img

        modify_btn.click(on_apply_colors,inputs=[color_checkboxes],outputs=final_img)

    return demo


# 启动 Gradio 界面
if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()