# ALYAi - Hybrid RAG AI Assistant

A personalized digital twin and school guide AI assistant built using a Hybrid Retrieval-Augmented Generation (RAG) architecture. It combines the conversational power of **Qwen2.5-1.5B** with the semantic search capabilities of **Sentence-Transformers**, all running completely offline.

**Author:** Alyssa H. Requillo  
**Course:** BS Information Technology  
**Institution:** Trinidad Municipal College (TMC)  

---

## 🌟 Key Features

* **100% Offline Execution:** After an initial model download, the AI runs entirely on your local machine without needing an internet connection.
* **Dynamic Semantic Search:** Uses `all-MiniLM-L6-v2` to understand the *meaning* of your questions, not just exact keywords.
* **Typo Correction Engine:** Automatically detects misspelled words, cross-references them against a dynamic vocabulary list, and asks for user confirmation before searching.
* **Adaptive Prompting:** Automatically detects if you are asking for official institutional mandates (and returns exact text) or general knowledge (allowing for natural conversational paraphrasing).
* **Smart Summarization:** Detects keywords like "brief" or "summarize" to automatically condense large blocks of information into short, digestible sentences.

---

## 📂 Project Structure

```text
ALYAi/
├── datasets/
│   └── train.txt          # The dynamic knowledge base (Topics & Descriptions)
├── test.py                # The main interactive chat script and RAG pipeline
├── requirements.txt       # Python dependencies list
├── .gitignore             # Prevents heavy system files from uploading to GitHub
└── README.md              # Project documentation

⚙️ Setup & Installation
Follow these steps to get the chatbot running on your local machine:

1. Clone the Repository
Bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
2. Set Up a Virtual Environment
It is highly recommended to run this in an isolated Python environment to avoid package conflicts.

On Windows:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
(Note: If Windows blocks the activation script, run Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass first).

On Mac/Linux:

Bash
python3 -m venv .venv
source .venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
🚀 How to Run the AI
To start the interactive chat interface, run:

Bash
python test.py
⚠️ Note on First Launch: On the very first run, the script will automatically download approximately 3.2 GB of open-source model weights directly into your system cache. All subsequent launches will load instantly and work completely offline.

💬 How to Use — Asking Questions
Because ALYAi uses semantic search, you do not need to memorize exact command phrases. You can talk to it naturally!

Examples of what you can ask:

"What is the TMC vision?" (Returns the exact official mandate).

"Give me a brief history of TMC." (Detects the word "brief" and summarizes the history paragraph).

"Who is Alyssa?" (Pulls from the Creator Profile).

Handling Typos:
If you type something incorrectly (e.g., "What is the misson?"), the AI will pause and ask:
AI: Did you mean: "what is the mission"? ("misson" -> "mission")

Just press Enter or type Y to proceed!

📝 How to Update the AI's Knowledge
Unlike traditional models, you do not need to retrain this AI.

To teach ALYAi new facts, simply open datasets/train.txt and add a new block of text using this format:

Plaintext
Topic: [The Subject Name]
Description: [A highly detailed paragraph explaining the subject.]
Save the file and restart test.py. The AI will instantly read the new text, vectorize it, and be ready to answer questions about it immediately!