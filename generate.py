"""Generate Shakespeare-style text from a trained checkpoint.

Usage:
    python generate.py --preset large --prompt "JULIET:" --length 500
"""

import argparse

import torch

from data import CharTokenizer
from model import GPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="large")
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--length", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(f"results/{args.preset}/model.pt", map_location=device, weights_only=True)
    cfg = checkpoint["config"]

    tokenizer = CharTokenizer("".join(checkpoint["vocab"]))
    model = GPT(cfg["n_layer"], cfg["n_head"], cfg["n_embd"], tokenizer.vocab_size, cfg["block_size"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    prompt = args.prompt.replace("\\n", "\n")
    idx = torch.tensor([tokenizer.encode(prompt)], device=device)
    out = model.generate(idx, args.length, temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
