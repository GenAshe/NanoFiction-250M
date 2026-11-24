import argparse
from pathlib import Path
import os
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast

def prepare_data(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train_bin = os.path.join(out_dir, "train.bin")
    val_bin = os.path.join(out_dir, "val.bin")
    if os.path.exists(train_bin) and os.path.exists(val_bin):
        return train_bin, val_bin

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    eot = tok.eos_token_id

    for split, path in [("train", train_bin), ("validation", val_bin)]:
        ds = load_dataset("deepmind/pg19", split=split)
        with open(path, "wb") as f:
            for ex in ds:
                ids = tok.encode(ex["text"])
                ids.append(eot)
                f.write(np.array(ids, dtype=np.uint16).tobytes())
    return train_bin, val_bin

def get_batch(path, block_size, batch_size, device):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = np.random.randint(0, len(data) - block_size - 1, size=(batch_size,))
    x = torch.from_numpy(np.stack([data[i:i+block_size].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([data[i+1:i+1+block_size].astype(np.int64) for i in ix]))
    if str(device).startswith("cuda"):
        x, y = x.pin_memory(), y.pin_memory()
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

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
        # PyTorch built-in flash attention
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
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
        # weight tying: input embedding and output projection share weights
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 trick)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        # subtract tied positional+output overlap is fine to ignore for reporting
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"sequence length {t} > block_size {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def train(data_dir="data", out_dir="out"):
    # ---- hyperparameters ----
    eval_interval = 500
    eval_iters = 20
    log_interval = 10
    save_interval = 1000

    batch_size = 64
    grad_accum_steps = 8           # effective batch = 64 * 8 * 1024 = 524k tokens/step
    block_size = 1024

    learning_rate = 2.5e-4
    max_iters = 5_000              # ~1.05 epochs over PG19 (~2.6B tokens seen)
    weight_decay = 0.1
    beta1, beta2 = 0.9, 0.95
    grad_clip = 1.0
    warmup_iters = 200
    lr_decay_iters = 5_000         # must match max_iters for cosine to bottom out
    min_lr = 2.5e-5                # 10% of peak lr

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    ptdtype = torch.bfloat16 if use_bf16 else torch.float32
    ctx = (torch.amp.autocast(device_type="cuda", dtype=ptdtype)
           if device == "cuda" else nullcontext())

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")  # use TF32 for fp32 matmuls

    # ---- data ----
    train_bin, val_bin = prepare_data(out_dir=data_dir)

    # ---- model ----
    config = GPTConfig(block_size=block_size, n_layer=16, n_head=16, n_embd=1024)
    model = GPT(config).to(device)
    print(f"[model] {model.num_params()/1e6:.1f}M parameters")

    try:
        model = torch.compile(model)
        print("[model] compiled with torch.compile")
    except Exception as e:
        print(f"[model] torch.compile unavailable: {e}")

    # ---- optimizer ----
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=(beta1, beta2),
        fused=(device == "cuda"),
    )

    def get_lr(it):
        if it < warmup_iters:
            return learning_rate * (it + 1) / (warmup_iters + 1)
        if it > lr_decay_iters:
            return min_lr
        ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split, path in [("train", train_bin), ("val", val_bin)]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(path, block_size, batch_size, device)
                with ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # ---- training loop ----
    t0 = time.time()
    X, Y = get_batch(train_bin, block_size, batch_size, device)
    for it in range(max_iters):
        lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if it % eval_interval == 0:
            losses = estimate_loss()
            print(f"[eval] step {it}: train {losses['train']:.4f}  val {losses['val']:.4f}")

        # forward + backward with gradient accumulation
        for _ in range(grad_accum_steps):
            with ctx:
                _, loss = model(X, Y)
                loss = loss / grad_accum_steps
            X, Y = get_batch(train_bin, block_size, batch_size, device)  # prefetch
            loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if it % log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"[train] step {it}: loss {loss.item()*grad_accum_steps:.4f}  "
                  f"lr {lr:.2e}  dt {dt*1000:.0f}ms")

        if it > 0 and it % save_interval == 0:
            torch.save(
                {"model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "config": config, "iter": it},
                os.path.join(out_dir, f"ckpt_{it}.pt"),
            )
            print(f"[ckpt] saved ckpt_{it}.pt")

    torch.save({"model": model.state_dict(), "config": config},
               os.path.join(out_dir, "final.pt"))
    print(f"[done] saved {out_dir}/final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain Nanofiction-250M on PG-19.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "out")
    args = parser.parse_args()
    train(data_dir=args.data_dir, out_dir=args.out_dir)
