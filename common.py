"""Shared helpers: load a frozen ESM-2 model and turn sequences into per-residue embeddings."""
import torch
from transformers import AutoTokenizer, EsmModel

# Small model (8M params, 320-dim) runs on a free Colab GPU or even CPU.
# Later try "facebook/esm2_t12_35M_UR50D" (480-dim) or "facebook/esm2_t30_150M_UR50D" (640-dim).
ESM_NAME = "facebook/esm2_t6_8M_UR50D"


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_esm(name=ESM_NAME, device=None):
    device = device or get_device()
    tok = AutoTokenizer.from_pretrained(name)
    model = EsmModel.from_pretrained(name).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return tok, model, device


@torch.no_grad()
def embed(seq, tok, model, device, max_len=512):
    """Return a (L, D) tensor of per-residue embeddings (CLS/EOS tokens removed)."""
    seq = seq[:max_len]
    enc = tok(seq, return_tensors="pt").to(device)
    h = model(**enc).last_hidden_state[0, 1:-1]
    return h.cpu()
