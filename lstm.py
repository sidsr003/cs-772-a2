"""
translit_seq2seq.py

Character-level LSTM encoder-decoder for Devanagari -> Roman transliteration.

Usage example:
python translit_seq2seq.py --data data/translit.jsonl --epochs 15 --batch-size 128 --save model.pt
"""

import argparse
import json
import random
from collections import Counter
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -----------------------
# Utility / Vocab classes
# -----------------------

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class CharVocab:
    def __init__(self, tokens=None, min_freq=1):
        self.min_freq = min_freq
        self.token2idx = {}
        self.idx2token = []
        if tokens:
            self.build(tokens)

    def build(self, tokens: List[str]):
        # tokens: iterator of characters across dataset
        freq = Counter(tokens)
        # start with specials
        self.idx2token = list(SPECIAL_TOKENS)
        for t in sorted([c for c, f in freq.items() if f >= self.min_freq]):
            if t in SPECIAL_TOKENS:
                continue
            self.idx2token.append(t)
        self.token2idx = {t: i for i, t in enumerate(self.idx2token)}

    def __len__(self):
        return len(self.idx2token)

    def encode(self, s: str) -> List[int]:
        out = []
        for ch in s:
            out.append(self.token2idx.get(ch, self.token2idx[UNK_TOKEN]))
        return out

    def decode(self, idxs: List[int]) -> str:
        chars = []
        for i in idxs:
            if i < 0 or i >= len(self.idx2token):
                chars.append(UNK_TOKEN)
            else:
                chars.append(self.idx2token[i])
        return "".join(chars)

    @property
    def pad_idx(self):
        return self.token2idx[PAD_TOKEN]

    @property
    def sos_idx(self):
        return self.token2idx[SOS_TOKEN]

    @property
    def eos_idx(self):
        return self.token2idx[EOS_TOKEN]

    @property
    def unk_idx(self):
        return self.token2idx[UNK_TOKEN]


# -----------------------
# Dataset + collate_fn
# -----------------------

class TransliterationDataset(Dataset):
    def __init__(self, records: List[Tuple[str, str]]):
        """
        records: list of (src_str, tgt_str)
        """
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def collate_fn(batch, src_vocab: CharVocab, tgt_vocab: CharVocab, device: torch.device):
    """
    batch: list of (src_str, tgt_str)
    returns:
      src_padded: (B, S)
      src_lengths: (B,)
      tgt_input_padded: (B, T)  # sos + target chars until before eos
      tgt_output_padded: (B, T) # target chars including eos (what to predict)
      tgt_lengths: (B,)
    """
    src_seqs = [s for s, t in batch]
    tgt_seqs = [t for s, t in batch]
    # encode
    src_idxs = [src_vocab.encode(s) for s in src_seqs]
    # for target we create input with SOS and output with EOS
    tgt_in_idxs = [[tgt_vocab.sos_idx] + tgt_vocab.encode(t) for t in tgt_seqs]
    tgt_out_idxs = [tgt_vocab.encode(t) + [tgt_vocab.eos_idx] for t in tgt_seqs]

    src_lengths = [len(x) for x in src_idxs]
    tgt_lengths = [len(x) for x in tgt_out_idxs]

    max_src = max(src_lengths)
    max_tgt = max(tgt_lengths)  # tgt_out length

    # pad
    pad = src_vocab.pad_idx
    src_padded = torch.full((len(batch), max_src), pad, dtype=torch.long)
    for i, seq in enumerate(src_idxs):
        src_padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    pad_t = tgt_vocab.pad_idx
    tgt_input_padded = torch.full((len(batch), max_tgt), pad_t, dtype=torch.long)
    tgt_output_padded = torch.full((len(batch), max_tgt), pad_t, dtype=torch.long)

    # Fill: tgt_in has length = 1 + len(original target chars) (no eos), tgt_out has len = len(original target chars)+1 (with eos)
    for i, (tin, tout) in enumerate(zip(tgt_in_idxs, tgt_out_idxs)):
        tgt_input_padded[i, : len(tin)] = torch.tensor(tin, dtype=torch.long)
        tgt_output_padded[i, : len(tout)] = torch.tensor(tout, dtype=torch.long)

    # move to device
    return (
        src_padded.to(device),
        torch.tensor(src_lengths, dtype=torch.long, device=device),
        tgt_input_padded.to(device),
        tgt_output_padded.to(device),
        torch.tensor(tgt_lengths, dtype=torch.long, device=device),
    )


# -----------------------
# Models
# -----------------------

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)
        self.n_layers = n_layers
        self.hid_dim = hid_dim

    def forward(self, src, src_lengths):
        # src: (B, S)
        embedded = self.embedding(src)  # (B, S, E)
        # pack
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # h_n: (num_layers, B, hid_dim)
        return out, (h_n, c_n)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=1, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.n_layers = n_layers
        self.hid_dim = hid_dim

    def forward(self, input_tokens, hidden):
        """
        input_tokens: (B, 1) token indices for current time step
        hidden: (h_n, c_n) each shape (n_layers, B, hid_dim)
        returns:
          output_logits: (B, output_dim)
          hidden: updated hidden state
        """
        emb = self.embedding(input_tokens)  # (B, 1, E)
        output, hidden = self.lstm(emb, hidden)  # output: (B, 1, hid_dim)
        logits = self.fc_out(output.squeeze(1))  # (B, output_dim)
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lengths, tgt_in, teacher_forcing_ratio=0.5):
        """
        src: (B, S)
        src_lengths: (B,)
        tgt_in: (B, T_in)  # contains SOS + target chars (no EOS)
        Returns logits: (B, T_out, V) where T_out = tgt_out length
        """
        batch_size = src.size(0)
        # Encode
        enc_out, (h_n, c_n) = self.encoder(src, src_lengths)
        # initialize decoder hidden with encoder's final hidden
        dec_hidden = (h_n, c_n)  # shapes compatible if encoder/decoder have same num_layers and hid_dim
        max_tgt_out_len = tgt_in.size(1)  # we will predict up to this length (we expect tgt_out has same length as tgt_in shifted)
        outputs = torch.zeros(batch_size, max_tgt_out_len, self.decoder.output_dim, device=self.device)

        # first input to decoder is the sos token already in tgt_in[:, 0]
        input_tok = tgt_in[:, 0].unsqueeze(1)  # (B,1)
        for t in range(0, max_tgt_out_len):
            logits, dec_hidden = self.decoder(input_tok, dec_hidden)
            outputs[:, t] = logits
            # determine next input: either teacher forcing or use predicted
            teacher_force = random.random() < teacher_forcing_ratio
            if teacher_force and t + 1 < tgt_in.size(1):
                next_input = tgt_in[:, t + 1].unsqueeze(1)  # teacher: feed ground-truth next token
            else:
                next_tokens = logits.argmax(dim=1).unsqueeze(1)
                next_input = next_tokens
            input_tok = next_input
        return outputs


# -----------------------
# Training / Inference
# -----------------------

def train_epoch(model: Seq2Seq, dataloader: DataLoader, optimizer, criterion, clip=1.0, tf_ratio=0.5):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader):
        src, src_lens, tgt_in, tgt_out, tgt_lens = batch
        optimizer.zero_grad()
        outputs = model(src, src_lens, tgt_in, teacher_forcing_ratio=tf_ratio)
        # outputs: (B, T_out, V)
        # tgt_out: (B, T_out)
        B, T, V = outputs.size()
        outputs_flat = outputs.view(B * T, V)
        tgt_flat = tgt_out.view(B * T)
        loss = criterion(outputs_flat, tgt_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model: Seq2Seq, dataloader: DataLoader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            src, src_lens, tgt_in, tgt_out, tgt_lens = batch
            outputs = model(src, src_lens, tgt_in, teacher_forcing_ratio=0.0)  # no teacher forcing
            B, T, V = outputs.size()
            loss = criterion(outputs.view(B * T, V), tgt_out.view(B * T))
            total_loss += loss.item()
    return total_loss / len(dataloader)


def greedy_decode(model: Seq2Seq, src_seq: str, src_vocab: CharVocab, tgt_vocab: CharVocab, max_len=100):
    model.eval()
    device = model.device
    with torch.no_grad():
        src_idx = src_vocab.encode(src_seq)
        src_tensor = torch.tensor([src_idx], dtype=torch.long, device=device)
        src_len = torch.tensor([len(src_idx)], dtype=torch.long, device=device)
        enc_out, (h_n, c_n) = model.encoder(src_tensor, src_len)
        hidden = (h_n, c_n)
        input_tok = torch.tensor([[tgt_vocab.sos_idx]], dtype=torch.long, device=device)
        out_idxs = []
        for _ in range(max_len):
            logits, hidden = model.decoder(input_tok, hidden)
            next_tok = logits.argmax(dim=1).item()
            if next_tok == tgt_vocab.eos_idx:
                break
            out_idxs.append(next_tok)
            input_tok = torch.tensor([[next_tok]], dtype=torch.long, device=device)
        return tgt_vocab.decode(out_idxs)
    

def beam_search_decode(model, src_seq, src_vocab, tgt_vocab, beam_size=3, max_len=80):
    model.eval()
    device = model.device
    with torch.no_grad():
        # Encode
        src_idx = torch.tensor([src_vocab.encode(src_seq)], dtype=torch.long, device=device)
        src_len = torch.tensor([len(src_idx[0])], dtype=torch.long, device=device)
        _, (h_n, c_n) = model.encoder(src_idx, src_len)
        hidden = (h_n, c_n)

        # Each beam: (log_prob, seq, hidden_state)
        beams = [(0.0, [tgt_vocab.sos_idx], (h_n, c_n))]
        completed = []

        for _ in range(max_len):
            new_beams = []

            for log_prob, seq, (h, c) in beams:
                input_tok = torch.tensor([[seq[-1]]], dtype=torch.long, device=device)
                logits, (new_h, new_c) = model.decoder(input_tok, (h, c))
                log_probs = torch.log_softmax(logits, dim=1)
                topk_logp, topk_idx = log_probs.topk(beam_size)

                for i in range(beam_size):
                    next_tok = topk_idx[0, i].item()
                    total_logp = log_prob + topk_logp[0, i].item()
                    new_seq = seq + [next_tok]
                    # Clone hidden state to detach from computation graph and beam interference
                    new_hidden = (new_h.clone(), new_c.clone())
                    if next_tok == tgt_vocab.eos_idx:
                        completed.append((total_logp, new_seq))
                    else:
                        new_beams.append((total_logp, new_seq, new_hidden))

            if not new_beams:
                break

            # Keep top-k active beams
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]

        # Choose best sequence
        if completed:
            best = max(completed, key=lambda x: x[0])
            seq = best[1][1:-1]  # drop SOS and EOS
        else:
            seq = beams[0][1][1:]  # drop SOS
        return tgt_vocab.decode(seq)



# -----------------------
# Data loading
# -----------------------

def read_jsonl(path: str) -> List[Tuple[str, str]]:
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            src = obj.get("native word") or ""
            tgt = obj.get("english word") or ""
            src = src.strip()
            tgt = tgt.strip().lower()
            if src and tgt:
                recs.append((src, tgt))
    return recs


def build_vocabs(records: List[Tuple[str, str]], min_freq=1):
    src_chars = []
    tgt_chars = []
    for s, t in records:
        for ch in s:
            src_chars.append(ch)
        for ch in t:
            tgt_chars.append(ch)
    src_vocab = CharVocab()
    src_vocab.build(src_chars)
    tgt_vocab = CharVocab()
    tgt_vocab.build(tgt_chars)
    return src_vocab, tgt_vocab


# -----------------------
# Main / CLI
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="lstm-model.pt")
    parser.add_argument("--save", type=str, default="lstm-model.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hid-dim", type=int, default=256)
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--train-split", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"Device: {device}")

    if args.mode == "train":
        raw_data = read_jsonl(args.data)
        records = [(s, t) for s, t in raw_data]
        random.shuffle(records)
        split = int(len(records) * args.train_split)
        train, val = records[:split], records[split:]
        src_vocab, tgt_vocab = build_vocabs(train)

        def collate_wrap(batch):
            return collate_fn(batch, src_vocab, tgt_vocab, device)

        train_loader = DataLoader(TransliterationDataset(train), batch_size=args.batch_size, shuffle=True, collate_fn=collate_wrap)
        val_loader = DataLoader(TransliterationDataset(val), batch_size=args.batch_size, shuffle=False, collate_fn=collate_wrap)

        encoder = Encoder(len(src_vocab), args.emb_dim, args.hid_dim)
        decoder = Decoder(len(tgt_vocab), args.emb_dim, args.hid_dim)
        model = Seq2Seq(encoder, decoder, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)

        best_loss = float("inf")
        for e in range(1, args.epochs + 1):
            tr_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_loss = evaluate(model, val_loader, criterion)
            print(f"Epoch {e}: train={tr_loss:.4f}, val={val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(
                    {"model_state": model.state_dict(), "src_vocab": src_vocab.idx2token, "tgt_vocab": tgt_vocab.idx2token},
                    args.save,
                )
                print("Saved best model to", args.save)

    elif args.mode == "predict":
        ckpt = torch.load(args.model, map_location=device)
        src_vocab = CharVocab()
        src_vocab.idx2token = ckpt["src_vocab"]
        src_vocab.token2idx = {t: i for i, t in enumerate(src_vocab.idx2token)}
        tgt_vocab = CharVocab()
        tgt_vocab.idx2token = ckpt["tgt_vocab"]
        tgt_vocab.token2idx = {t: i for i, t in enumerate(tgt_vocab.idx2token)}

        encoder = Encoder(len(src_vocab), 128, 256)
        decoder = Decoder(len(tgt_vocab), 128, 256)
        model = Seq2Seq(encoder, decoder, device).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        data = read_jsonl(args.data)
        greedy_out, beam_out = [], []

        for src, tgt in tqdm(data, desc="Transliterating", unit="word"):
            greedy_pred = greedy_decode(model, src, src_vocab, tgt_vocab)
            beam_pred = beam_search_decode(model, src, src_vocab, tgt_vocab, beam_size=args.beam_size)

            greedy_out.append({
                "hindi": src,
                "pred": greedy_pred,
                "gold": tgt
            })
            beam_out.append({
                "hindi": src,
                "pred": beam_pred,
                "gold": tgt
            })

        with open("lstm_predictions/output_greedy.jsonl", "w", encoding="utf-8") as fg:
            for x in greedy_out:
                fg.write(json.dumps(x, ensure_ascii=False) + "\n")
        with open("lstm_predictions/output_beam.jsonl", "w", encoding="utf-8") as fb:
            for x in beam_out:
                fb.write(json.dumps(x, ensure_ascii=False) + "\n")

        print("Inference complete. Saved lstm_predictions/output_greedy.jsonl and lstm_predictions/output_beam.jsonl")


if __name__ == "__main__":
    main()