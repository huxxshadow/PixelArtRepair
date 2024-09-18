import gradio as gr
from color_quantization import (
    kmeans_quantization,
    median_cut_quantization,
    floyd_steinberg_dithering,
    median_cut_perceptual_weighting
)
from pixelation import pixelate_image, mosaic_pixelation, oil_paint_pixelation, hierarchical_pixelation
import math


# Define the pixelation and color quantization processing function
def pixelate_and_quantize(image, pixel_size, n_colors, methods, interpolation, pixelation_types):
    results = []

    # Perform pixelation based on the user's selected pixelation method
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
            raise ValueError(f"Unknown pixelation method: {pixelation_type}")

        # Perform color quantization based on the user's selected method and add the quantized image to the list
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
                raise ValueError(f"Unknown quantization method: {method}")

            # Return only the quantized images
            results.append(quantized_img)

    return results


# Gradio user interface
def gradio_interface():
    with gr.Blocks(title="Image Pixelation Tool") as demo:
        gr.Markdown("## Image Pixelation Tool (AI Pixel Repair)")
        gr.Markdown("""
                This tool provides a variety of image pixelation effects and color quantization algorithms, 
                including classic nearest neighbor pixelation, mosaic effect, oil painting effect, and hierarchical pixelation.
                Users can choose different pixelation methods and perform color quantization on the image.

                ### License Agreement
                This tool is released under the **MPL 2.0 (Mozilla Public License 2.0)** license. The license allows modification, distribution, 
                and commercial use, as long as the modified part of the source code remains open-source.

                - **Allowed**: Modification, distribution, commercial use.
                - **Restriction**: Modified source code must remain open-source.

                [Read the full MPL 2.0 License](https://www.mozilla.org/en-US/MPL/2.0/)

                **Note: Please contact the author for further authorization before using this tool for commercial purposes.**

                Contact: huxxshadowhzy@gmail.com

                Copyright © 2024
                """)
        with gr.Row():
            gr.Image(value="Example.png", show_download_button=True,interactive=False,show_fullscreen_button=False)
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

            with gr.Column():
                pixel_size_slider = gr.Slider(minimum=2, maximum=100, value=10, step=1, label="Choose Pixel Block Size")
                color_slider = gr.Slider(minimum=2, maximum=128, value=16, step=1, label="Choose Number of Colors")

                method_checkboxes = gr.CheckboxGroup(
                    choices=["K-Means", "Median Cut", "Floyd-Steinberg Dithering", "MedianCut(PerceptualWeighting)"],
                    value=["K-Means"],
                    label="Select Color Quantization Methods"
                )
                pixelation_checkboxes = gr.CheckboxGroup(
                    choices=["Classic Nearest Neighbor", "Mosaic", "Oil Painting", "Hierarchical"],
                    value=["Classic Nearest Neighbor"],
                    label="Select Pixelation Methods"
                )
                interpolation_radio = gr.Radio(
                    choices=["Nearest", "Bilinear", "Bicubic", "Lanczos"],
                    value="Nearest",
                    label="Choose Interpolation Method (Only for Classic Nearest Neighbor)"
                )

        btn = gr.Button("Generate Pixelated Image")

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
                                # Calculate the current pixelation and quantization method
                                current_pixelation = pixelation_types[idx // len(methods)]
                                current_method = methods[idx % len(methods)]
                                # Update the image label with pixelation and color quantization methods
                                label = f"Pixelation: {current_pixelation}, Color Quantization: {current_method}"
                                gr.Image(images[idx], label=label, format="png")
                                idx += 1
        return demo


# Launch Gradio interface
if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()