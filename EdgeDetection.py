import cv2
import numpy as np
import gradio as gr

def neighborhood_average(image, mask, kernel_size=3):
    """
    自定义邻域平均函数，用于修复图像中需要修复的区域
    image: 原始图像
    mask: 掩膜，白色部分表示需要修复的区域
    kernel_size: 用于计算邻域平均的窗口大小
    """
    # 创建图像副本
    repaired_image = image.copy()

    # 获取图像的高度和宽度
    height, width, _ = image.shape

    # 遍历掩膜图像的每个像素
    for y in range(height):
        for x in range(width):
            # 如果掩膜的当前像素为白色（255），表示需要修复
            if mask[y, x] == 255:
                # 获取邻域的窗口范围
                y_min = max(0, y - kernel_size // 2)
                y_max = min(height, y + kernel_size // 2 + 1)
                x_min = max(0, x - kernel_size // 2)
                x_max = min(width, x + kernel_size // 2 + 1)

                # 取出邻域窗口
                neighborhood = image[y_min:y_max, x_min:x_max]

                # 计算邻域的平均值
                mean_color = np.mean(neighborhood, axis=(0, 1))

                # 将平均颜色填充到修复图像的对应位置
                repaired_image[y, x] = mean_color

    return repaired_image

def repair_edges(image, low_threshold=50, high_threshold=150, kernel_size=3, neighborhood_size=3):
    # 将图像转换为灰度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Canny算法检测图像边缘
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)

    # 创建形态学内核
    morph_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    #
    # # 使用形态学操作修复边缘
    edges_dilated = cv2.dilate(edges, morph_kernel, iterations=1)
    edges_eroded = cv2.erode(edges_dilated, morph_kernel, iterations=1)



    # 将处理后的边缘作为掩膜
    mask = edges_eroded

    # 使用自定义的邻域平均方法对图像进行修复
    repaired_image = neighborhood_average(image, mask, neighborhood_size)

    # 返回修复后的原始图像和生成的掩膜
    return repaired_image, mask

# 创建Gradio接口
interface = gr.Interface(
    fn=repair_edges,  # 处理函数
    inputs=[
        gr.Image(type="numpy", label="上传图片"),
        gr.Slider(0, 255, value=50, label="低阈值"),
        gr.Slider(0, 255, value=150, label="高阈值"),
        gr.Slider(1, 10, value=3, step=1, label="修复内核大小"),
        gr.Slider(1, 11, value=3, step=1, label="邻域平均窗口大小")
    ],
    outputs=[
        gr.Image(type="numpy", label="修复后的图像"),
        gr.Image(type="numpy", label="掩膜图像")
    ],
    title="Canny边缘修复并使用邻域平均",
    description="使用Canny算法检测图像边缘并修复原图的边缘毛刺，使用自定义邻域平均法修复损坏的区域。"
)

# 启动Gradio应用
interface.launch()