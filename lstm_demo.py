# lstm_demo.py
import json
import torch
import torch.nn as nn
from lstm import Encoder, Decoder, Seq2Seq, CharVocab, greedy_decode, beam_search_decode

import gradio as gr

# -----------------------
# Load model and vocabs
# -----------------------

MODEL_PATH = "lstm-model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ckpt = torch.load(MODEL_PATH, map_location=device)

# Load vocabs
src_vocab = CharVocab()
src_vocab.idx2token = ckpt["src_vocab"]
src_vocab.token2idx = {t: i for i, t in enumerate(src_vocab.idx2token)}

tgt_vocab = CharVocab()
tgt_vocab.idx2token = ckpt["tgt_vocab"]
tgt_vocab.token2idx = {t: i for i, t in enumerate(tgt_vocab.idx2token)}

# Model config
EMB_DIM = 128
HID_DIM = 256

encoder = Encoder(len(src_vocab), EMB_DIM, HID_DIM)
decoder = Decoder(len(tgt_vocab), EMB_DIM, HID_DIM)
model = Seq2Seq(encoder, decoder, device).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# -----------------------
# Transliteration function
# -----------------------

def transliterate(word: str, method: str = "greedy", beam_size: int = 3) -> str:
    word = word.strip()
    if not word:
        return ""
    if method == "greedy":
        return greedy_decode(model, word, src_vocab, tgt_vocab)
    elif method == "beam":
        return beam_search_decode(model, word, src_vocab, tgt_vocab, beam_size=beam_size)
    else:
        return "Invalid decoding method"

# -----------------------
# Gradio interface
# -----------------------

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Hindi (Devanagari) → Roman Transliteration
        Enter a Hindi word and choose a decoding method.
        """
    )

    with gr.Row():
        inp_word = gr.Textbox(label="Hindi word", placeholder="हिंदी शब्द")
        method = gr.Radio(choices=["greedy", "beam"], label="Decoding method", value="greedy")
        beam_slider = gr.Slider(minimum=2, maximum=10, step=1, value=3, label="Beam size (if beam decoding)")

    out_word = gr.Textbox(label="Predicted transliteration")

    def wrapper(word, method, beam_size):
        return transliterate(word, method, beam_size)

    btn = gr.Button("Transliterate")
    btn.click(wrapper, inputs=[inp_word, method, beam_slider], outputs=out_word)

demo.launch()
