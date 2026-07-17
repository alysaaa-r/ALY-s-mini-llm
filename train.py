import os
import torch
from tokenizer import WordTokenizer
from model import MiniLLM

# =========================
# Hyperparameters
# =========================
n_embd = 64          # size of each word's vector
n_head = 4           # number of parallel attention heads (head_size = n_embd / n_head = 16)
n_layer = 4          # number of stacked Transformer Blocks
block_size = 96       # max tokens the model can look at (longest Q+A pair was 74 tokens)
batch_size = 8        # how many examples trained on at once
learning_rate = 3e-4
max_iters = 3000
eval_interval = 200

torch.manual_seed(1337)

# =========================
# Load and tokenize the dataset
# =========================
with open("datasets/train.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = WordTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Total tokens: {len(data)}")

# train / validation split (90% train, 10% validation)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Train tokens: {len(train_data)} | Validation tokens: {len(val_data)}")


# Find every position where a Q&A pair starts (the token id for "topic",
# since each block now begins with a "Topic: X" line) so training windows
# are aligned to real example boundaries, not random offsets.
topic_token_id = tokenizer.stoi["topic"]
all_boundaries = [i for i, t in enumerate(data.tolist()) if t == topic_token_id]

train_boundaries = [i for i in all_boundaries if i < n and i + block_size <= n]
val_boundaries = [i for i in all_boundaries if i >= n and i + block_size <= len(data)]

# Fallback: if the validation split ends up with no usable boundaries
# (can happen with a very small dataset), reuse train boundaries for eval too.
if not val_boundaries:
    val_boundaries = train_boundaries


def get_batch(split):
    boundaries = train_boundaries if split == "train" else val_boundaries
    starts = [boundaries[torch.randint(len(boundaries), (1,)).item()] for _ in range(batch_size)]
    x = torch.stack([data[s:s + block_size] for s in starts])
    y = torch.stack([data[s + 1:s + block_size + 1] for s in starts])
    return x, y


@torch.no_grad()
def estimate_loss(model, eval_iters=20):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# =========================
# Create model + optimizer
# =========================
model = MiniLLM(
    vocab_size=tokenizer.vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    n_head=n_head,
    n_layer=n_layer,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# =========================
# Training loop
# =========================
print(f"\nStarting training for {max_iters} iterations...\n")

for step in range(max_iters):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {step:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

print("\nTraining completed.")

# =========================
# Save checkpoint
# =========================
os.makedirs("checkpoint", exist_ok=True)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "stoi": tokenizer.stoi,
    "itos": tokenizer.itos,
    "vocab_size": tokenizer.vocab_size,
    "config": {
        "n_embd": n_embd,
        "block_size": block_size,
        "n_head": n_head,
        "n_layer": n_layer,
    },
}

torch.save(checkpoint, "checkpoint/mini-llm.pt")
print("Saved checkpoint to checkpoint/mini-llm.pt")

# =========================
# Quick sample generation
# =========================
prompt = "Question: What is the vision of the school?\nAnswer:"
encoded = tokenizer.encode(prompt)
context = torch.tensor([encoded], dtype=torch.long)  # shape (1, T)

generated = model.generate(context, max_new_tokens=40, temperature=0.7)
print("\n=== SAMPLE OUTPUT ===")
print(tokenizer.decode(generated[0].tolist()))