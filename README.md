# Nanofiction-250M

A small decoder-only language model pretrained from random initialization on PG-19.
The GPT-2/nanoGPT-style model uses 16 layers, 16 attention heads, width 1024, tied
token/output embeddings, and a 1024-token context window. It is a text-completion
experiment, not an instruction-tuned assistant. The reported run saw approximately
2.6 billion tokens over 5,000 optimizer steps.

## Setup and inference

Install a PyTorch build appropriate for your CPU or GPU, then:

```sh
python -m pip install -r requirements.txt
python Inference.py --ckpt weights/Nanofiction-250M.safetensors "A storm brewed beyond the window" --tokens 150
```

Download the supplied safetensors checkpoint separately and place it in `weights/`.
Omit the prompt for interactive mode; `/quit` exits. CPU inference is supported but
slow. GPU dtype is selected automatically; `--device cpu --dtype fp32` forces CPU.
The model is custom PyTorch code, not a Transformers AutoModel checkpoint.

## Training and export

```sh
python TrainNanofiction250M.py --data-dir data --out-dir out
python ExportSafetensors.py out/final.pt weights/Nanofiction-250M.safetensors --trust-checkpoint
```

Training downloads PG-19 and the GPT-2 tokenizer. Its defaults target a large-memory
GPU (batch 64, accumulation 8); edit the hyperparameters for smaller hardware.
Dataset-library compatibility and GPU execution have not been validated in this
publication-preparation environment. Export accepts only your own trusted pickle
checkpoints; inference uses safetensors. Large datasets and intermediate checkpoints
are intentionally excluded from this repository.

## Limitations and attribution

The 1024-token context limits long-range consistency; generations can be incorrect,
repetitive, biased, or inappropriate. No general-purpose benchmark superiority is claimed.
Architecture and implementation draw on Andrej Karpathy's nanoGPT and GPT-2 materials;
PG-19 is from Google DeepMind and tokenization uses the GPT-2 tokenizer.
See `ATTRIBUTION.md`. A distribution license has not yet been selected for this cleaned
code package; establish the applicable upstream notices before public release.
