# transformer_demo.py
"""
Gradio demo for Transformer-based Hindi→Roman Transliteration
"""

import json
import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Optional, List
import gradio as gr

# ========================================
# Model Classes
# ========================================

@dataclass
class TransformerConfig:
    d_model: int = 256
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    local_window: Optional[int] = None


class CharVocab:
    def __init__(self):
        self.idx2token: List[str] = []
        self.token2idx: dict = {}
    
    @property
    def pad_idx(self) -> int:
        return self.token2idx.get("<pad>", 0)
    
    @property
    def sos_idx(self) -> int:
        return self.token2idx.get("<sos>", 1)
    
    @property
    def eos_idx(self) -> int:
        return self.token2idx.get("<eos>", 2)
    
    @property
    def unk_idx(self) -> int:
        return self.token2idx.get("<unk>", 3)
    
    def encode(self, text: str) -> List[int]:
        return [self.token2idx.get(ch, self.unk_idx) for ch in text]
    
    def decode(self, ids: List[int]) -> str:
        chars = []
        for idx in ids:
            if idx < 0 or idx >= len(self.idx2token):
                chars.append("<unk>")
            else:
                chars.append(self.idx2token[idx])
        return "".join(chars)
    
    def __len__(self):
        return len(self.idx2token)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, maxlen: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        position = torch.arange(0, maxlen, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(maxlen, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LocalTransformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, config: TransformerConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        
        self.src_tok_emb = nn.Embedding(src_vocab_size, config.d_model, padding_idx=0)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, config.d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(config.d_model, dropout=config.dropout)
        self.pos_decoder = PositionalEncoding(config.d_model, dropout=config.dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)
        self.generator = nn.Linear(config.d_model, tgt_vocab_size)
    
    def build_encoder_mask(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        window = self.config.local_window
        if window is None or window <= 0 or window >= seq_len:
            return None
        
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=torch.float32)
        for i in range(seq_len):
            start = max(0, i - window)
            end = min(seq_len, i + window + 1)
            mask[i, start:end] = 0.0
        return mask
    
    def build_decoder_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        window = self.config.local_window
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=torch.float32), diagonal=1)
        
        if window is not None and window > 0 and window < seq_len:
            for i in range(seq_len):
                start = max(0, i - window)
                if start > 0:
                    mask[i, :start] = float('-inf')
        return mask
    
    def encode(self, src, src_key_padding_mask):
        src_emb = self.src_tok_emb(src) * math.sqrt(self.config.d_model)
        src_emb = self.pos_encoder(src_emb)
        src_mask = self.build_encoder_mask(src_emb.size(1), src_emb.device)
        return self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    
    def decode(self, tgt, memory, src_key_padding_mask, tgt_key_padding_mask):
        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.config.d_model)
        tgt_emb = self.pos_decoder(tgt_emb)
        tgt_mask = self.build_decoder_mask(tgt_emb.size(1), tgt_emb.device)
        out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.generator(out)


# ========================================
# Decoding Functions
# ========================================

def greedy_decode(model: LocalTransformer, src_seq: str, src_vocab: CharVocab, tgt_vocab: CharVocab, max_len: int = 80) -> str:
    """Greedy decoding"""
    model.eval()
    device = model.device
    
    with torch.no_grad():
        src_idxs = src_vocab.encode(src_seq)
        if not src_idxs:
            return ""
        
        src_tensor = torch.tensor([src_idxs], dtype=torch.long, device=device)
        src_key_padding_mask = (src_tensor == src_vocab.pad_idx)
        memory = model.encode(src_tensor, src_key_padding_mask)
        
        ys = torch.tensor([[tgt_vocab.sos_idx]], dtype=torch.long, device=device)
        output_tokens = []
        
        for _ in range(max_len):
            tgt_key_padding_mask = (ys == tgt_vocab.pad_idx)
            logits = model.decode(ys, memory, src_key_padding_mask, tgt_key_padding_mask)
            next_token = logits[:, -1, :].argmax(dim=-1).item()
            
            if next_token == tgt_vocab.eos_idx:
                break
            
            output_tokens.append(next_token)
            ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)
        
        return tgt_vocab.decode(output_tokens)


def beam_search_decode(model: LocalTransformer, src_seq: str, src_vocab: CharVocab, tgt_vocab: CharVocab, 
                       beam_size: int = 5, max_len: int = 80) -> str:
    """Beam search decoding"""
    model.eval()
    device = model.device
    
    with torch.no_grad():
        src_idxs = src_vocab.encode(src_seq)
        if not src_idxs:
            return ""
        
        src_tensor = torch.tensor([src_idxs], dtype=torch.long, device=device)
        src_key_padding_mask = (src_tensor == src_vocab.pad_idx)
        memory = model.encode(src_tensor, src_key_padding_mask)
        
        beams = [(0.0, [tgt_vocab.sos_idx])]
        completed = []
        
        for _ in range(max_len):
            new_beams = []
            
            for log_prob, seq in beams:
                if seq[-1] == tgt_vocab.eos_idx:
                    completed.append((log_prob, seq))
                    continue
                
                ys = torch.tensor([seq], dtype=torch.long, device=device)
                tgt_key_padding_mask = (ys == tgt_vocab.pad_idx)
                logits = model.decode(ys, memory, src_key_padding_mask, tgt_key_padding_mask)
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
                
                top_log_p, top_idx = log_probs.topk(beam_size)
                
                for lp, idx in zip(top_log_p[0], top_idx[0]):
                    new_seq = seq + [idx.item()]
                    new_beams.append((log_prob + lp.item(), new_seq))
            
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_size]
            
            if all(seq[-1] == tgt_vocab.eos_idx for _, seq in beams):
                completed.extend(beams)
                break
        
        completed.extend(beams)
        
        if not completed:
            best_seq = [tgt_vocab.sos_idx, tgt_vocab.eos_idx]
        else:
            _, best_seq = max(completed, key=lambda x: x[0] / len(x[1]))
        
        decoded = []
        for tok in best_seq[1:]:
            if tok == tgt_vocab.eos_idx:
                break
            decoded.append(tok)
        
        return tgt_vocab.decode(decoded)


# ========================================
# Load Model
# ========================================

MODEL_PATH = "transformer_model.pt"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

print(f"Using device: {device}")
print("Loading model checkpoint...")

# Load checkpoint
ckpt = torch.load(MODEL_PATH, map_location=device)

# Check if vocabularies are in checkpoint
if 'src_vocab' not in ckpt or 'tgt_vocab' not in ckpt:
    print("\n" + "="*70)
    print("ERROR: Vocabularies not found in checkpoint!")
    print("="*70)
    print("\nPlease re-save your model with vocabularies using this code in your notebook:\n")
    print("""
def vocab_to_dict(vocab):
    return {
        'idx2token': list(vocab.idx2token),
        'token2idx': dict(vocab.token2idx)
    }

torch.save({
    'model_state': model.state_dict(),
    'config': config.__dict__,
    'src_vocab': vocab_to_dict(src_vocab),
    'tgt_vocab': vocab_to_dict(tgt_vocab),
    'history': history,
}, 'transformer-model.pt')
    """)
    print("="*70)
    exit(1)

# Load vocabularies
src_vocab = CharVocab()
src_vocab.idx2token = ckpt["src_vocab"]["idx2token"]
src_vocab.token2idx = ckpt["src_vocab"]["token2idx"]

tgt_vocab = CharVocab()
tgt_vocab.idx2token = ckpt["tgt_vocab"]["idx2token"]
tgt_vocab.token2idx = ckpt["tgt_vocab"]["token2idx"]

print(f"Source vocab size: {len(src_vocab)}")
print(f"Target vocab size: {len(tgt_vocab)}")

# Load config
config_dict = ckpt["config"]
config = TransformerConfig(**config_dict)

# Create and load model
model = LocalTransformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    config=config,
    device=device
).to(device)

model.load_state_dict(ckpt["model_state"])
model.eval()

print("✅ Model loaded successfully!")


# ========================================
# Transliteration Function
# ========================================

def transliterate(word: str, method: str = "greedy", beam_size: int = 5) -> str:
    """Transliterate a Hindi word to Roman script"""
    word = word.strip()
    if not word:
        return ""
    
    if method == "greedy":
        return greedy_decode(model, word, src_vocab, tgt_vocab)
    elif method == "beam":
        return beam_search_decode(model, word, src_vocab, tgt_vocab, beam_size=beam_size)
    else:
        return "Invalid decoding method"


# ========================================
# Gradio Interface
# ========================================

examples = [
    ["भारत", "greedy", 5],
    ["नमस्ते", "greedy", 5],
    ["दिल्ली", "beam", 5],
    ["मुंबई", "beam", 5],
    ["स्वागत", "beam", 10],
]

with gr.Blocks(title="Hindi→Roman Transliteration") as demo:
    gr.Markdown(
        """
        # 🔤 Hindi (Devanagari) → Roman Transliteration
        
        Enter a Hindi word in Devanagari script and choose a decoding method.
        
        **Decoding Methods:**
        - **Greedy**: Fast, selects the most likely character at each step
        - **Beam Search**: Slower but more accurate, explores multiple possibilities
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            inp_word = gr.Textbox(
                label="Hindi Word (Devanagari)",
                placeholder="हिंदी शब्द यहाँ लिखें...",
                lines=1
            )
            
            with gr.Row():
                method = gr.Radio(
                    choices=["greedy", "beam"],
                    label="Decoding Method",
                    value="greedy"
                )
                beam_slider = gr.Slider(
                    minimum=2,
                    maximum=10,
                    step=1,
                    value=5,
                    label="Beam Size (for beam search)",
                    info="Higher values = more accurate but slower"
                )
            
            btn = gr.Button("Transliterate", variant="primary")
        
        with gr.Column(scale=2):
            out_word = gr.Textbox(
                label="Transliterated Output (Roman)",
                lines=1,
                interactive=False
            )
            
            gr.Markdown(
                """
                ### Quick Tips:
                - Greedy decoding is faster for real-time use
                - Beam search (width 5-10) gives better quality
                - Works best with common Hindi words
                """
            )
    
    gr.Markdown("### Example Words:")
    gr.Examples(
        examples=examples,
        inputs=[inp_word, method, beam_slider],
        outputs=out_word,
        fn=transliterate,
        cache_examples=False
    )
    
    btn.click(
        fn=transliterate,
        inputs=[inp_word, method, beam_slider],
        outputs=out_word
    )
    
    inp_word.submit(
        fn=transliterate,
        inputs=[inp_word, method, beam_slider],
        outputs=out_word
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
