from logging import PlaceHolder
import gradio as gr

from transformers import pipeline

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")




def translate(text):
    print (pipe(text)[0]["translation_text"])
    return pipe(text)[0]["translation_text"]

with open('tranduccion.txt') as f:
    contents = f.read()
    print(contents)


with gr.Blocks() as demo2:
    gr.Markdown(
        """
    # Hello World!
    Start typing below to see the output.
    """)
    inp = gr.Textbox(value=contents)
    out = gr.Textbox()
    inp.change(translate, inp, out)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            english = gr.Textbox(label="English text")
            translate_btn = gr.Button(value="Translate")
        with gr.Column():
            espanol = gr.Textbox(label="Spanish Text")

    translate_btn.click(translate, inputs=english, outputs=espanol)



demo.launch(debug=True)
