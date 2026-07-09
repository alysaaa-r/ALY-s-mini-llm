import re
import torch
from tokenizer import WordTokenizer
from model import MiniLLM

MAX_NEW_TOKENS = 60
TEMPERATURE = 0.6
MATCH_THRESHOLD = 0.5   # how similar the user's question must be to a known one (0-1)

# =========================
# Load checkpoint
# =========================
checkpoint = torch.load("checkpoint/mini-llm.pt", map_location="cpu")

tokenizer = WordTokenizer.from_vocab(checkpoint["stoi"], checkpoint["itos"])

config = checkpoint["config"]
model = MiniLLM(
    vocab_size=checkpoint["vocab_size"],
    n_embd=config["n_embd"],
    block_size=config["block_size"],
    n_head=config["n_head"],
    n_layer=config["n_layer"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded. Type a question, or 'exit' to quit.\n")


# =========================
# Exact-match safety net
# =========================
def _normalize_words(text: str):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def load_qa_pairs(path="datasets/train.txt"):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    pairs = []
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    for block in blocks:
        q_match = re.search(r"Question:\s*(.+)", block)
        a_match = re.search(r"Answer:\s*(.+)", block, re.DOTALL)
        if q_match and a_match:
            question = q_match.group(1).strip()
            answer = a_match.group(1).strip()
            pairs.append((set(_normalize_words(question)), question, answer))
    return pairs


qa_pairs = load_qa_pairs()


def find_best_match(user_input: str):
    user_words = set(_normalize_words(user_input))
    if not user_words:
        return None, 0.0

    best_score = 0.0
    best_answer = None
    for q_words, _, answer in qa_pairs:
        if not q_words:
            continue
        overlap = len(user_words & q_words)
        score = (2 * overlap) / (len(user_words) + len(q_words))  # Dice coefficient
        if score > best_score:
            best_score = score
            best_answer = answer

    return best_answer, best_score


# =========================
# Generative fallback
# =========================
def generate_answer(question: str) -> str:
    prompt = f"Question: {question}\nAnswer:"

    try:
        encoded = tokenizer.encode(prompt)
    except KeyError as e:
        return f"(I don't recognize the word {e} — it wasn't in my training data.)"

    if not encoded:
        return "(I couldn't understand any words in that question.)"

    context = torch.tensor([encoded], dtype=torch.long)  # shape (1, T)

    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)

    full_text = tokenizer.decode(generated[0].tolist())

    prompt_decoded = tokenizer.decode(encoded)
    if full_text.startswith(prompt_decoded):
        answer = full_text[len(prompt_decoded):].strip()
    else:
        answer = full_text.strip()

    cut_point = answer.find(" question ")
    if cut_point != -1:
        answer = answer[:cut_point].strip()

    return answer if answer else "(No answer generated.)"


def answer_question(question: str) -> str:
    matched_answer, score = find_best_match(question)
    if matched_answer is not None and score >= MATCH_THRESHOLD:
        return matched_answer   # guaranteed-correct, properly-cased answer

    return generate_answer(question)   # fall back to real model generation


# =========================
# Chat loop
# =========================
while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    answer = answer_question(user_input)
    print("AI:", answer, "\n")