# ALYAi - Hybrid RAG AI Assistant

A personalized digital twin and school guide AI assistant built using a **Hybrid Retrieval-Augmented Generation (RAG)** architecture. It combines the conversational power of **Qwen2.5-1.5B** with the semantic search capabilities of **Sentence-Transformers**, all running completely offline.

**Author:** Alyssa H. Requillo  
**Course:** BS Information Technology  
**Institution:** Trinidad Municipal College (TMC)

---

# 🌟 Key Features

- **100% Offline Execution**  
  After the initial model download, ALYAi runs entirely on your local machine without requiring an internet connection.

- **Dynamic Semantic Search**  
  Uses the `all-MiniLM-L6-v2` embedding model to understand the *meaning* of questions instead of relying on exact keyword matching.

- **Typo Correction Engine**  
  Automatically detects misspelled words, compares them against a dynamically generated vocabulary, and asks for confirmation before searching.

- **Adaptive Prompting**  
  Detects whether the user is asking for official institutional information or general knowledge, ensuring official mandates are returned verbatim while conversational responses remain natural.

- **Smart Summarization**  
  Automatically summarizes lengthy information whenever words like **"brief"**, **"summarize"**, or similar requests are detected.

---

# 📂 Project Structure

```text
ALYAi/
├── datasets/
│   └── train.txt          # Dynamic knowledge base (Topics & Descriptions)
├── test.py                # Main chatbot and Hybrid RAG pipeline
├── requirements.txt       # Python dependencies
├── .gitignore             # Prevents unnecessary files from being uploaded
└── README.md              # Project documentation
```

---

# ⚙️ Setup & Installation

Follow these steps to run ALYAi on your local computer.

## 1. Clone the Repository

```bash
git clone https://github.com/alysaaa-r/ALY-s-mini-llm.git
cd ALY-s-mini-llm
```

---

## 2. Create a Virtual Environment

Using a virtual environment is highly recommended to avoid dependency conflicts.

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> **Note:** If PowerShell blocks the activation script, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Running ALYAi

Launch the chatbot with:

```bash
python test.py
```

---

## ⚠️ First Launch

On the first execution, ALYAi automatically downloads approximately **3.2 GB** of open-source model weights into your local Hugging Face cache.

This happens **only once**.

After the download finishes:

- ✅ No internet connection is required.
- ✅ Everything runs locally.
- ✅ Future launches load the models directly from your computer.

---

# 💬 Asking Questions

ALYAi uses **semantic search**, so you can ask questions naturally without memorizing exact keywords.

## Examples

```text
What is the TMC vision?
```

Returns the official Vision statement.

```text
Give me a brief history of TMC.
```

Detects the word **"brief"** and summarizes the history.

```text
Who is Alyssa?
```

Retrieves information from the Creator Profile.

---

## Handling Typos

If a word is misspelled, ALYAi will ask for confirmation.

Example:

```text
User:
What is the misson?

AI:
Did you mean:
"What is the mission"?
("misson" → "mission")
```

Simply press **Enter** or type:

```text
Y
```

to continue with the corrected search.

---

# 📝 Updating ALYAi's Knowledge

Unlike traditional AI models, **no retraining is required**.

Simply open:

```text
datasets/train.txt
```

Add a new knowledge entry using the following format:

```text
Topic: Subject Name

Description:
A detailed paragraph explaining the subject.
```

Example:

```text
Topic: Campus Library

Description:
The Trinidad Municipal College Library provides books,
computers, study spaces, and online learning resources
for students and faculty.
```

Save the file and restart the chatbot:

```bash
python test.py
```

ALYAi will automatically:

1. Read the updated dataset.
2. Generate new vector embeddings.
3. Make the new information immediately searchable.

No model retraining is necessary.

---

# 🛠 Technologies Used

- **Python**
- **Qwen2.5-1.5B-Instruct**
- **Sentence-Transformers**
- **all-MiniLM-L6-v2**
- **PyTorch**
- **Transformers (Hugging Face)**
- **NumPy**
- **Scikit-learn**

---

# 📌 Summary

ALYAi is a fully offline Hybrid Retrieval-Augmented Generation (RAG) assistant designed to serve as a personalized digital twin and campus information guide for Trinidad Municipal College. By combining semantic search, adaptive prompting, typo correction, and intelligent summarization, it delivers accurate, context-aware responses while remaining easy to update through a simple text-based knowledge base.