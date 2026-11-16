"""Train the character-level GPT on Shakespeare.

Usage:
    python train.py --preset course   # the small configuration from the original assignment
    python train.py --preset large    # a bigger model that produces readable text
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from data import load_datasets
from model import GPT

PRESETS = {
    "course": dict(block_size=10, n_layer=3, n_head=3, n_embd=48, dropout=0.0, lr=3e-4, batch_size=64, epochs=10),
    "large": dict(block_size=128, n_layer=6, n_head=8, n_embd=256, dropout=0.1, lr=5e-4, batch_size=64, epochs=10),
}
STEPS_PER_EPOCH = 1500
EVAL_BATCHES = 150
PROMPT = "ROMEO:\n"


def evaluate(model, loader, criterion, device, vocab_size):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= EVAL_BATCHES:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits.view(-1, vocab_size), y.view(-1)).item()
            correct += (logits.argmax(-1) == y).sum().item()
            count += y.numel()
    model.train()
    return total_loss / min(EVAL_BATCHES, len(loader)), correct / count


def save_curves(history, out_path: Path, title: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, history["train_loss"], marker="o", label="Train")
    ax1.plot(epochs, history["test_loss"], marker="o", label="Test")
    ax1.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["train_acc"], marker="o", label="Train")
    ax2.plot(epochs, history["test_acc"], marker="o", label="Test")
    ax2.set(xlabel="Epoch", ylabel="Next-character accuracy", title="Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=PRESETS, default="large")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = dict(PRESETS[args.preset])
    if args.epochs:
        cfg["epochs"] = args.epochs
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, train_set, test_set = load_datasets("data/train_shakespeare.txt", "data/test_shakespeare.txt", cfg["block_size"])
    vocab_size = tokenizer.vocab_size
    model = GPT(cfg["n_layer"], cfg["n_head"], cfg["n_embd"], vocab_size, cfg["block_size"], cfg["dropout"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Preset '{args.preset}': {n_params / 1e6:.2f}M parameters on {device}")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        sampler=RandomSampler(train_set, replacement=True, num_samples=STEPS_PER_EPOCH * cfg["batch_size"]),
    )
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"] * STEPS_PER_EPOCH)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": [], "samples": []}
    start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running_loss, correct, count = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            correct += (logits.argmax(-1) == y).sum().item()
            count += y.numel()

        test_loss, test_acc = evaluate(model, test_loader, criterion, device, vocab_size)
        history["train_loss"].append(running_loss / len(train_loader))
        history["train_acc"].append(correct / count)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        prompt = torch.tensor([tokenizer.encode(PROMPT)], device=device)
        model.eval()
        sample = tokenizer.decode(model.generate(prompt, 200, temperature=0.8, top_k=10)[0].tolist())
        history["samples"].append(sample)
        print(f"Epoch {epoch}: train loss {history['train_loss'][-1]:.3f} | test loss {test_loss:.3f} | test acc {test_acc:.3f}")
        print(sample, "\n")

    elapsed = time.time() - start
    out_dir = Path("results") / args.preset
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "vocab": tokenizer.vocab}, out_dir / "model.pt")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "params": n_params, "train_seconds": round(elapsed), **history}, f, indent=2)
    Path("assets").mkdir(exist_ok=True)
    save_curves(history, Path("assets") / f"curves_{args.preset}.png", f"Character-level GPT - '{args.preset}' preset ({n_params / 1e6:.2f}M params)")
    print(f"Done in {elapsed / 60:.1f} min. Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
