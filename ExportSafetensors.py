"""Export a trusted local Nanofiction training checkpoint to safetensors."""
import argparse
from pathlib import Path
import torch
from safetensors.torch import save_file
from TrainNanofiction250M import GPTConfig

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--trust-checkpoint", action="store_true", help="Allow pickle loading of your own training checkpoint.")
    args = ap.parse_args()
    if not args.trust_checkpoint:
        ap.error("Only export checkpoints you trust; pass --trust-checkpoint to acknowledge pickle loading.")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {k.removeprefix("_orig_mod."): v.detach().cpu().contiguous().clone() for k,v in ck["model"].items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(args.output))

if __name__ == "__main__":
    main()
