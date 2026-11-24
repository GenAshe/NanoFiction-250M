from pathlib import Path
import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import GPT2TokenizerFast

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 16
    n_head: int = 16
    n_embd: int = 1024
    dropout: float = 0.0
    bias: bool = False

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head, self.n_embd = config.n_head, config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Re-tie weights (the safetensors file stores both, but we want them shared
        # in memory for inference efficiency).
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.config.block_size, f"sequence length {t} > block_size {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.9, top_k=200, eos_id=None, stream_cb=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            logits[:, 50257:] = -float("inf")
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if stream_cb is not None:
                stream_cb(int(idx_next.item()))
            if eos_id is not None and int(idx_next.item()) == eos_id:
                break
        return idx

def load_model(safetensors_path: str, device: str, dtype: torch.dtype):
    config = GPTConfig()
    model = GPT(config)
    state = load_file(safetensors_path, device="cpu")
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, config

def generate_once(model, tokenizer, prompt, max_new_tokens, temperature, top_k, device, stream=True):
    ids = tokenizer.encode(prompt)
    if len(ids) == 0:
        ids = [tokenizer.eos_token_id]
    x = torch.tensor([ids], device=device, dtype=torch.long)

    print(prompt, end="", flush=True)

    if stream:
        generated = []
        printed = ""

        def cb(tok_id):
            nonlocal printed
            generated.append(tok_id)
            text = tokenizer.decode(generated)
            new_text = text[len(printed):]
            print(new_text, end="", flush=True)
            printed = text

        out = model.generate(
            x, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k,
            eos_id=tokenizer.eos_token_id, stream_cb=cb,
        )
    else:
        out = model.generate(
            x, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k,
            eos_id=tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(out[0, len(ids):].tolist())
        print(completion, end="", flush=True)

    print()
    return out


def repl(model, tokenizer, device, max_new_tokens, temperature, top_k):
    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        if prompt == "/quit":
            break
        if prompt.startswith("/temp "):
            temperature = float(prompt.split()[1]); continue
        if prompt.startswith("/topk "):
            top_k = int(prompt.split()[1]); continue
        if prompt.startswith("/tokens "):
            max_new_tokens = int(prompt.split()[1]); continue
        generate_once(model, tokenizer, prompt, max_new_tokens, temperature, top_k, device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default=None,
                    help="Prompt text. If omitted, drops into an interactive REPL.")
    ap.add_argument("--ckpt", default=str(Path(__file__).resolve().parent / "weights" / "Nanofiction-250M.safetensors"))
    ap.add_argument("--tokens", type=int, default=300, help="max new tokens")
    ap.add_argument("--temp", type=float, default=0.9, help="sampling temperature")
    ap.add_argument("--top_k", type=int, default=200, help="top-k sampling (0 disables)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None, help="cuda | cpu (auto-detect if omitted)")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--no-stream", action="store_true", help="disable token streaming")
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.dtype == "auto":
        if device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif device == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    model, _ = load_model(args.ckpt, device=device, dtype=dtype)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    top_k = args.top_k if args.top_k and args.top_k > 0 else None

    if args.prompt is None:
        repl(model, tokenizer, device, args.tokens, args.temp, top_k)
        return

    generate_once(
        model, tokenizer, args.prompt,
        max_new_tokens=args.tokens,
        temperature=args.temp,
        top_k=top_k,
        device=device,
        stream=not args.no_stream,
    )

if __name__ == "__main__":
    main()
