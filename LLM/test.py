from vertexai.generative_models import GenerativeModel, Image
import json
import pandas as pd
import gradio as gr


def generate_text(img) -> str:
    multimodal_model = GenerativeModel("gemini-pro-vision")

    image = Image.load_from_file(img)

    response = multimodal_model.generate_content(
        [
            image,
            f"""Act like a text scanner. Extract text as it is without analyzing it and without summarizing it. Treat all images as a whole document and analyze them accordingly. Think of it as a document with multiple pages, each image being a page. Understand page-to-page flow logically and semantically.""",
        ]
    )
    return response.text


with gr.Blocks(title="Title") as demo:
    with gr.Row():
        with gr.Column():
            output = gr.Textbox(label="Output")
        img = gr.Image(type="filepath")
    submit = gr.Button("Extract")
    submit.click(fn=generate_text, inputs=[img], outputs=output)

demo.launch()
