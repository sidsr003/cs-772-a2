import os
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

# Load your API key
load_dotenv(".env")
client = OpenAI()

MODEL_NAME = "gpt-4.1"  # or gpt-4o

def transliterate_text(hindi_text):
    """Send a single Hindi input to GPT for transliteration"""
    if not hindi_text.strip():
        return "Please enter some Hindi text."

    prompt = (
        "You are a transliteration assistant.\n"
        "Transliterate the following Hindi (Devanagari) text into English (Roman script), "
        "preserving pronunciation.\n"
        "Do not use accents or diacritics.\n"
        "Return only the transliteration.\n\n"
        f"Hindi: {hindi_text}\nEnglish:"
    )

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.output_text.strip()
    except Exception as e:
        return f"Error: {e}"

# Gradio interface
demo = gr.Interface(
    fn=transliterate_text,
    inputs=gr.Textbox(
        label="Enter Hindi Text (Devanagari)",
        placeholder="उदाहरण: नमस्ते दुनिया",
        lines=3
    ),
    outputs=gr.Textbox(label="English Transliteration"),
    title="🪶 Hindi → English Transliteration",
    description="Type Hindi (Devanagari) words and get their English transliteration using GPT."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)