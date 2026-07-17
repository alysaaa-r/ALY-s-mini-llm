import re
import difflib
import torch
from tokenizer import WordTokenizer
from model import MiniLLM

MAX_NEW_TOKENS = 60
TEMPERATURE = 0.6
MATCH_THRESHOLD = 0.45   # how similar the user's question must be to a known one (0-1)
TYPO_CUTOFF = 0.8         # how close a misspelled word must be to a real vocab word (0-1)

# Common "filler" words that don't carry topic meaning — ignored when scoring
# how similar a question is to a known one, so phrasing style (command vs
# question, polite vs blunt) doesn't matter as much as the actual topic.
STOPWORDS = {
    "i", "me", "my", "you", "your", "yours", "we", "us", "our",
    "please", "give", "tell", "share", "want", "know", "could", "would",
    "can", "do", "does", "did", "is", "are", "was", "were", "be",
    "the", "a", "an", "of", "to", "in", "on", "for", "about", "with",
    "what", "who", "when", "where", "how", "that", "this", "it",
}

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
# Small talk (handled outside the model entirely — like a real agent's
# hardcoded intents for things that aren't really "questions")
# =========================
SMALL_TALK = [
    (["hi", "hello", "hey", "greetings"], "Hello! Ask me about TMC's vision, mission, goal, or about me."),
    (["thank", "thanks", "thankyou"], "You're welcome!"),
    (["bye", "goodbye", "seeya"], "Goodbye! Type 'exit' anytime to quit."),
    (["how are you"], "I'm just a small language model, but I'm running well! Ask me something about TMC."),
]


def check_small_talk(text: str):
    lowered = text.lower().strip().rstrip("!.?")
    for triggers, response in SMALL_TALK:
        for trigger in triggers:
            if lowered == trigger or lowered.startswith(trigger + " "):
                return response
    return None


# =========================
# Exact-match safety net (stopword-filtered)
# =========================
def _normalize_words(text: str):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _content_words(words):
    """Strip out filler words, keeping only topic-carrying words.
    Falls back to the full word set if everything happened to be a stopword."""
    content = set(w for w in words if w not in STOPWORDS)
    return content if content else set(words)


def load_qa_pairs(path="datasets/train.txt"):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    pairs = []
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    for block in blocks:
        topic_match = re.search(r"Topic:\s*(\S+)", block)
        q_match = re.search(r"Question:\s*(.+)", block)
        a_match = re.search(r"Answer:\s*(.+)", block, re.DOTALL)
        if q_match and a_match:
            question = q_match.group(1).strip()
            answer = a_match.group(1).strip()
            q_content = _content_words(_normalize_words(question))
            if topic_match:
                q_content = q_content | {topic_match.group(1).strip().lower()}
            pairs.append((q_content, question, answer))
    return pairs


qa_pairs = load_qa_pairs()

# Rarity weighting: words that appear in many different training questions
# (like "school", "tmc") are common and not very distinctive. Words that
# appear in only one or two questions (like "vision", "birthday") are
# highly distinctive. We weigh rare words more heavily so a match based
# only on a common word (e.g. just "school") isn't treated as confident.
_doc_freq = {}
for q_words, _, _ in qa_pairs:
    for w in q_words:
        _doc_freq[w] = _doc_freq.get(w, 0) + 1


def _word_weight(word):
    df = _doc_freq.get(word, 1)
    return 1.0 / df


def find_best_match(user_input: str):
    user_words = _content_words(_normalize_words(user_input))
    if not user_words:
        return None, 0.0

    best_score = 0.0
    best_answer = None
    for q_words, _, answer in qa_pairs:
        if not q_words:
            continue
        shared = user_words & q_words
        shared_weight = sum(_word_weight(w) for w in shared)
        total_weight = sum(_word_weight(w) for w in user_words) + sum(_word_weight(w) for w in q_words)
        score = (2 * shared_weight) / total_weight if total_weight > 0 else 0.0
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


# =========================
# Typo detection and correction
# =========================
VOCAB_WORDS = list(tokenizer.stoi.keys())


def suggest_correction(word: str):
    """Find the closest real vocabulary word to a possibly-misspelled word."""
    if word in tokenizer.stoi or word in STOPWORDS:
        return None  # already a real word, or a common word we don't want to "correct"

    matches = difflib.get_close_matches(word, VOCAB_WORDS, n=1, cutoff=TYPO_CUTOFF)
    return matches[0] if matches else None


def check_for_typos(question: str):
    words = _normalize_words(question)
    corrected_words = []
    corrections = []

    for word in words:
        suggestion = suggest_correction(word)
        if suggestion:
            corrected_words.append(suggestion)
            corrections.append((word, suggestion))
        else:
            corrected_words.append(word)

    corrected_question = " ".join(corrected_words)
    return corrected_question, corrections


# =========================
# Main answer routing
# =========================
def answer_question(user_input: str) -> str:
    # 1. Try the user's original phrasing first — command-style, polite,
    #    or blunt phrasing all get a fair shot before we assume it's a typo.
    matched_answer, score = find_best_match(user_input)
    if matched_answer is not None and score >= MATCH_THRESHOLD:
        return matched_answer

    # 2. No good match — check if unrecognized words might be typos, and
    #    ask the user to confirm before assuming a correction.
    corrected_question, corrections = check_for_typos(user_input)
    if corrections:
        correction_list = ", ".join(f'"{orig}" -> "{fix}"' for orig, fix in corrections)
        print(f'AI: Did you mean: "{corrected_question}"? ({correction_list})')
        confirm = input("    [Y/n]: ").strip().lower()

        if confirm in ("", "y", "yes"):
            matched_answer, score = find_best_match(corrected_question)
            if matched_answer is not None and score >= MATCH_THRESHOLD:
                return matched_answer
            return generate_answer(corrected_question)

    # 3. Still nothing — fall back to the real trained model.
    return generate_answer(user_input)


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

    small_talk_reply = check_small_talk(user_input)
    if small_talk_reply:
        print("AI:", small_talk_reply, "\n")
        continue

    answer = answer_question(user_input)
    print("AI:", answer, "\n")