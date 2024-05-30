from vertexai.preview.generative_models import GenerativeModel, Image
import json
import pandas as pd
import gradio as gr
import json
import gradio as gr


def generate_text(field1, field2, field3, img) -> str:
    multimodal_model = GenerativeModel("gemini-pro-vision")

    image = Image.load_from_file(img)

    response = multimodal_model.generate_content(
        [
            image,
            f"""Extract{field},{field2},{field3} from the image in the json schema:""",
        ]
    )
    return response.text


with gr.Blocks(title="Title") as demo:
    with gr.Row():
        with gr.Column():
            field1 = gr.Textbox(
                label="Field 1",
            )
            field2 = gr.Textbox(
                label="Field 2",
            )
            field3 = gr.Textbox(
                label="Field 3",
            )
            output = gr.Textbox(label="Output")
        img = gr.Image(type="filepath")
    submit = gr.Button("Extract")
    submit.click(fn=upload_image, inputs=[field1, field2, field3, img], outputs=output)

demo.launch()

test = generate_text(img="/pages/IMG_4172.JPG")
print(test)
