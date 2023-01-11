import gradio as gr
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')

def generate(text):
    result = generator(text, max_length=30, num_return_sequences=1)
    return result[0]["generated_text"]

examples = [
    ["The Moon's orbit around Earth has"],
    ["The smooth Borealis basin in the Northern Hemisphere covers 40%"],
]

gr.Interface(generate, "textbox", "text", examples=examples).launch()