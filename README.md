# Mini LLM — TMC Vision/Mission/Goal Q&A Assistant

A small, word-level Transformer language model built entirely from scratch in
PyTorch, trained to answer questions about Trinidad Municipal College's
Vision, Mission, Goal, Philosophy, and Slogan — as well as questions about
the author. Wrapped with a conversational layer that handles typos, casual
phrasing, and small talk, similar to how a real support agent would.

**Author:** Alyssa H. Requillo
**Course:** BS Information Technology, 4th Year
**Institution:** Trinidad Municipal College (TMC)

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup & Installation](#setup--installation)
3. [How to Use — Asking Questions](#how-to-use--asking-questions)
4. [How to Train](#how-to-train)
5. [Step-by-Step: How This Was Built From Scratch](#step-by-step-how-this-was-built-from-scratch)
6. [Terminal Commands Reference](#terminal-commands-reference)
7. [Known Limitations](#known-limitations)

---

## Project Structure

```
ALYai/
├── dataset/
│   └── train.txt          # 43 Topic/Question/Answer training triples
├── checkpoint/
│   └── mini-llm.pt         # Saved trained model weights + vocabulary
├── tokenizer.py             # Word-level tokenizer (text <-> numbers)
├── model.py                 # The Transformer architecture
├── train.py                 # Trains the model on dataset/train.txt
├── test.py                  # Interactive chat script
└── .venv/                   # Python virtual environment
```

---

## Setup & Installation

### Prerequisites
- **VS Code** — https://code.visualstudio.com/
- **Python 3.10+** — https://www.python.org/downloads/ (check "Add python.exe to PATH" during install)
- **Python extension for VS Code** (Extensions panel, search "Python")

### One-time setup (Windows PowerShell)

```powershell
# Create a virtual environment
py -m venv .venv

# Allow the activation script to run
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate it
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install torch numpy
```

You'll know the virtual environment is active when you see `(.venv)` at the
start of your terminal prompt. Every time you open a new terminal, re-run the
activate step before running any script.

---

## How to Use — Asking Questions

Once a trained checkpoint exists (`checkpoint/mini-llm.pt`), run:

```powershell
python test.py
```

You'll see:
```
Model loaded. Type a question, or 'exit' to quit.

You:
```

Type any question and press Enter. Type `exit` to quit.

### How answering works internally

`test.py` routes every message through up to **four layers**, in order,
mimicking how a real conversational agent handles a mix of small talk,
well-formed questions, and typos:

1. **Small talk** — greetings ("hi"), thanks, "how are you," and goodbyes are
   recognized directly by simple rules and answered instantly, without
   touching the trained model at all.
2. **Exact-match safety net** — your question is compared against all 43
   known training questions. Matching ignores filler/command words ("please,"
   "give," "tell me," "I want to know") and weighs **rare, topic-specific
   words more heavily than common ones** (e.g. "vision" counts far more than
   "school," since "school" appears in almost every question). If the best
   match is strong enough, the exact, correctly-worded stored answer is
   returned immediately — guaranteed accurate.
3. **Typo correction** — if no strong match is found, every unrecognized word
   is checked against the model's vocabulary for a close spelling match
   (e.g. "misson" → "mission"). If any are found, you're asked to confirm:
   ```
   AI: Did you mean: "what is the mission of the school"? ("misson" -> "mission")
       [Y/n]:
   ```
   Confirming re-runs the exact-match check on the corrected text.
4. **Generative fallback** — if nothing above resolves it, the question is
   fed into the actual trained neural network, which generates a response
   word-by-word based on patterns it learned during training. This is the
   only layer that can be imperfect, and is only reached for genuinely novel
   phrasing.

### Good example questions to try

**About the school:**
```
What is the vision of the school?
What is TMC's mission?
What are TMC's goals?
What is the philosophy of the school?
What is the slogan of the school?
```

**About the author:**
```
Who are you?
Introduce yourself.
What is your name?
What course are you taking?
What year level are you in?
When were you born?
How old are you?
Where do you study?
What are your interests?
```

**Command-style / conversational phrasing (also works):**
```
Give me the TMC vision
Can you tell me the mission of the school
I want to know the goal of TMC
Please share the slogan
Hi / Thank you / How are you / bye
```

**Deliberate typos (also works, with confirmation):**
```
What is the misson of the school?
How olde are you?
What is your naem?
```

### Tips for best results

- Phrase questions similarly to the examples above for the most reliable
  results — this style is what both the exact-match layer and the model
  itself were trained on.
- If a word is completely outside the training vocabulary and doesn't
  resemble any known word closely enough, you'll see:
  ```
  (I don't recognize the word 'xyz' — it wasn't in my training data.)
  ```

---

## How to Train

If you edit `dataset/train.txt` (add more Q&A pairs, fix wording, etc.),
retrain the model so `checkpoint/mini-llm.pt` reflects the changes:

```powershell
python train.py
```

This will:
1. Load and tokenize `dataset/train.txt`
2. Train for 3000 iterations, printing train/validation loss every 200 steps
3. Save the trained weights to `checkpoint/mini-llm.pt`
4. Print one sample generation at the end as a sanity check

Training takes a few minutes on a CPU. Watch the **train loss** — it should
drop from around 5.3 (random guessing) down toward 0.0–0.5 by the end,
confirming the model is learning your data.

**Note:** `test.py`'s exact-match and typo-correction layers read
`dataset/train.txt` directly and don't require retraining to update — only
changes that affect the trained model's actual generative behavior (editing
answers, adding new topics, changing `model.py`) require running
`train.py` again.

---

## Step-by-Step: How This Was Built From Scratch

This project was built piece by piece, testing each concept in isolation
before assembling the final files. Below is the full development story.

### Phase 1 — Preparing the Training Data

**Workflow used:**
1. Gathered the exact, official source text (Vision, Mission, Goal,
   Philosophy, Slogan; and personal info).
2. Brainstormed multiple phrasings for each question.
3. Paired every question with its correct answer.
4. Formatted consistently as:
   ```
   Topic: <topic name>
   Question: <question>
   Answer: <answer>
   ```
   with a blank line between each triple, so the model can learn to
   recognize the pattern.
5. Repeated each answer under several phrasings, since a small model learns
   from repeated exposure to the same pattern.

**Result:** 43 Topic/Question/Answer triples across 14 topics (Vision,
Mission, Goal, Philosophy, Slogan, Identity, Creator, Name, Course, Year,
Birthday, Age, School, Interests).

### Phase 2 — Building the Tokenizer (`tokenizer.py`)

Neural networks only understand numbers, not raw text, so a tokenizer
converts text → numbers (encoding) and numbers → text (decoding).

**Design decision — word-level, not character-level:** whole words carry
meaning directly, so the model needs less data to learn useful patterns than
a character-level model would.

**Design decision — lowercase, punctuation stripped:** so "School" and
"school" are treated as the same token, and question marks/periods don't
bloat the vocabulary with noise.

**Build steps, tested individually:**
1. **Splitting text into words** with a regular expression:
   ```python
   words = re.findall(r"[a-zA-Z0-9]+", text.lower())
   ```
2. **Building the vocabulary** — every unique word gets a number:
   ```python
   unique_words = sorted(set(words))
   stoi = {word: i for i, word in enumerate(unique_words)}   # string to index
   itos = {i: word for i, word in enumerate(unique_words)}   # index to string
   ```
3. **Encoding / decoding**:
   ```python
   def encode(text): return [stoi[w] for w in split_words(text)]
   def decode(ids):  return " ".join(itos[i] for i in ids)
   ```
4. **`from_vocab()` classmethod** — added so `test.py` can rebuild a
   tokenizer from a *saved* vocabulary (from the checkpoint) instead of
   rebuilding it from raw text every time.

**Verified result:** vocabulary built correctly from `dataset/train.txt`;
encode → decode round-trips matched the original text.

### Phase 3 — Building the Model (`model.py`)

A small **decoder-only Transformer** — the same architecture family as
GPT-style models, scaled down to run on a CPU.

**3.1 Token Embeddings** — converts each word ID into a learnable vector of
numbers (`nn.Embedding`), since a raw ID number has no meaningful
relationship to other IDs.

**3.2 Position Embeddings** — a second embedding table giving each *position*
in the sequence its own vector, added to the token embedding, so the model
knows word order (not just word identity).

**3.3 Self-Attention (Query, Key, Value)** — lets each word look at other
words and pull in relevant context:
- **Query** = what this word is looking for
- **Key** = what each word offers
- **Value** = the actual content pulled in on a match
```python
wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)
```

**3.4 Causal Mask** — blocks a word from seeing words that come *after* it
(since the model predicts left-to-right), using a lower-triangular mask that
sets "future" scores to `-inf` before softmax.

**3.5 Softmax + Weighted Value Blending** — turns scores into probabilities
that sum to 1, then uses them to compute a weighted average of Value vectors
— the new, context-aware representation of each word.

**3.6 Multi-Head Attention** — runs several attention heads in parallel (each
with independent Query/Key/Value weights), so the model can track multiple
kinds of relationships simultaneously. Outputs are concatenated and passed
through a final linear layer (`proj`) to blend information across heads.

**3.7 Feed-Forward Layer** — a small 2-layer network (expand 4x → ReLU →
shrink back) that processes each word's representation further after
attention.

**3.8 Transformer Block** — bundles Multi-Head Attention + Feed-Forward with:
- **Residual connections** (`x = x + ...`) — a shortcut path that keeps deep
  networks trainable by preventing gradients from vanishing.
- **LayerNorm** — re-centers and re-scales values before each sub-layer,
  keeping training numerically stable.

**3.9 Full Model Assembly (`MiniLLM`)** — embeddings → stack of Blocks →
final LayerNorm → output layer (`lm_head`) mapping to a probability score
for every word in the vocabulary.

**3.10 Batch Support** — the model processes a whole *batch* of sequences at
once (`(B, T)` instead of just `(T,)`), which is what allows efficient
training on multiple examples simultaneously.

**3.11 `generate()` method** — produces new text by repeatedly:
1. Predicting a probability distribution over the next word
2. Sampling one word from it (controlled by a `temperature` setting —
   lower = safer/more repetitive, higher = more random/creative)
3. Appending it to the sequence and repeating

### Phase 4 — Training (`train.py`)

**The core training loop:**
```python
for step in range(max_iters):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```
- **`get_batch()`** randomly samples chunks of tokenized text and creates
  `(input, target)` pairs, where the target is the same chunk shifted one
  word to the right — "given these words, predict the next one."
- **`loss.backward()`** (backpropagation) calculates how much each of the
  model's ~227,000 weights contributed to the error.
- **`optimizer.step()`** (AdamW) applies small adjustments to reduce that
  error next time.

**Bug found and fixed #1 — boundary-aligned batching:**
Initial training used fully random start positions for each training chunk.
Since `block_size` (96 tokens) was larger than the gap between Q&A pairs,
each training window often spanned *multiple* unrelated pairs at random,
unaligned offsets. This caused the model to partly learn shortcut patterns
based on position rather than actual question content, leading to incorrect
answers (e.g. answering "Who are you?" with the Vision statement).

**The fix:** `get_batch()` was changed to always start each training window
exactly at a boundary marker. This was refined twice:
- First aligned to the `"question"` token.
- Later upgraded to align to the `"topic"` token instead, once `Topic:`
  tags were added ahead of each `Question:` line — giving an even cleaner,
  unambiguous start-of-example signal.

**Training results after the fix:**
```
step    0 | train loss ~5.30 | val loss ~5.23   (random-guessing baseline)
step 2999 | train loss ~0.04 | val loss ~6-7     (heavily memorized)
```

The low train loss confirms the model successfully memorized the training
patterns — appropriate for this project's goal of reliably reproducing
official statements, rather than open-ended generalization.

### Phase 5 — Testing & Making It Behave Like a Real Agent (`test.py`)

Testing revealed several realistic conversational failure modes, each
addressed in turn:

**Problem 1 — personal-info sub-topics confused with each other.**
Institutional questions (vision, mission, goal) were reliably answered, but
similar personal questions (e.g. "What is your birthday?" vs "How old are
you?") were sometimes mixed up — a real limitation of training a small model
on a small, generically-worded dataset.

*Fixes applied, in order:*
1. Expanded personal info from 9 to 22 examples with more phrasing variety.
2. Rewrote answers to "echo" the question's keyword (e.g. *"My birthday is
   May 16, 2005..."*), helping attention "copy" the relevant pattern.
3. Added a **hybrid exact-match + generative fallback**:
   ```python
   def answer_question(question):
       matched_answer, score = find_best_match(question)
       if score >= MATCH_THRESHOLD:
           return matched_answer          # guaranteed-correct
       return generate_answer(question)   # real model generation
   ```
4. Added explicit **`Topic:` tags** to every training pair (Vision, Mission,
   Birthday, Age, etc.) — folded into the matching signal as a strong,
   low-frequency anchor word per topic, and used in training itself for
   cleaner boundary alignment (see Phase 4).

**Problem 2 — command-style and conversational phrasing failed.**
Questions like *"Give me the TMC vision"* or *"I want to know the goal of
TMC"* initially scored too low to match, since filler words ("give," "i,"
"want," "know") diluted the word-overlap score, and a naive fix
(stopword-removal alone) caused a *new* problem: generic shared words like
"school" or "tmc" could tie across multiple unrelated topics.

*Final fix — rarity-weighted matching:* instead of counting all words
equally, each word is weighted by how rare it is across the 43 training
questions (`weight = 1 / document_frequency`). Common words like "school"
(appearing in ~20+ questions) count very little; rare, topic-specific words
like "vision" or "birthday" count heavily. This is a lightweight version of
the **TF-IDF** technique used in real search/retrieval systems.

**Problem 3 — false-positive typo correction.**
An early typo-correction feature (using `difflib.get_close_matches`) once
"corrected" the real word **"want"** into **"what"**, since they're
similar-looking strings — corrupting valid input. The fix was reordering the
pipeline: exact-match is now always tried on the *original* wording first;
typo-correction is only offered as a fallback if that fails, and stopwords
are excluded from ever being "corrected."

**Problem 4 — no handling for greetings/small talk.**
Real conversations include things that aren't really "questions" at all
("hi," "thanks," "how are you"). These are now caught by a simple rule-based
layer *before* anything touches the trained model or dataset — the same
approach many production chatbots use for non-informational chit-chat.

**Final verified result:** 21/21 test questions across all categories
(institutional, personal, command-style, typo, small talk, and gibberish
rejection) answered correctly.

---

## Terminal Commands Reference

| Purpose | Command |
|---|---|
| Create virtual environment | `py -m venv .venv` |
| Allow venv activation (Windows) | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| Activate virtual environment | `.\.venv\Scripts\Activate.ps1` |
| Deactivate virtual environment | `deactivate` |
| Upgrade pip | `python -m pip install --upgrade pip` |
| Install dependencies | `pip install torch numpy` |
| Train the model | `python train.py` |
| Chat with the trained model | `python test.py` |

---

## Known Limitations

- **Vocabulary is fixed at training time.** Any word not present in
  `dataset/train.txt`, and not close enough to a known word for typo
  correction to catch, cannot be understood.
- **Generative fallback is the least reliable layer.** For genuinely novel
  questions that don't match anything in the exact-match layer, the raw
  generative model can occasionally produce imperfect or blended answers —
  this is a known, explainable limitation of training a small model on a
  small dataset, not a bug.
- **Not a general-purpose chatbot.** This model is intentionally narrow and
  memorization-focused — it is designed to reliably answer a fixed set of
  questions about TMC's Vision/Mission/Goal and the author, not to reason
  about arbitrary new topics.
- **Typo correction has a similarity cutoff.** Very garbled input (e.g.
  random keyboard mashing) is correctly left unrecognized rather than
  forcing a bad guess.
- **Small talk is rule-based, not learned.** Greetings/thanks/goodbye are
  matched by simple hardcoded triggers, not by the neural network — this is
  intentional (real agents commonly do this too), but it means unusual
  greeting phrasing may not be recognized.

---

*Built from scratch as a learning project to understand how Transformer-based
language models work internally — from token embeddings to a working,
trainable neural network, wrapped in a practical, agent-like conversational
layer.*