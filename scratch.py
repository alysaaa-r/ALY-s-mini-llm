import torch
from tokenizer import WordTokenizer
from model import MiniLLM

block_size = 16
batch_size = 4
n_embd = 32
n_head = 4
n_layer = 2

with open("datasets/train.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = WordTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x, y


model = MiniLLM(tokenizer.vocab_size, n_embd, block_size, n_head, n_layer)

xb, yb = get_batch("train")
logits, loss = model(xb, yb)   # full batch this time, not xb[0]

print("Batch input shape:", xb.shape)
print("Logits shape (full batch):", logits.shape)
print("Loss:", loss.item())

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print("Loss BEFORE training step:", loss.item())

optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()

# Run the model again on a NEW batch to see if loss changed
xb2, yb2 = get_batch("train")
logits2, loss2 = model(xb2, yb2)
print("Loss AFTER one training step:", loss2.item())

max_iters = 2000
eval_interval = 200

for step in range(max_iters):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0 or step == max_iters - 1:
        print(f"step {step:4d} | loss {loss.item():.4f}")

import os

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

context = torch.zeros((1, 1), dtype=torch.long)  # start with token id 0
generated = model.generate(context, max_new_tokens=20, temperature=0.8)
print("\nSample generation:")
print(tokenizer.decode(generated[0].tolist()))