import os
import re
import difflib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# TYPO THRESHOLD: How close a misspelled word must be to a real dataset word (0-1)
TYPO_CUTOFF = 0.68       

print("Loading models (this might take a moment)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
model_name = "Qwen/Qwen2.5-1.5B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)

# =====================================================================
# LOAD DATASET (WITH TOPIC TRACKING)
# =====================================================================
def load_and_vectorize_dataset(path="datasets/train.txt"):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Build vocabulary list for typo detection
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    vocab_words = list(set(words))
    
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    dataset_records = []
    questions_only = []
    
    for block in blocks:
        topic_match = re.search(r"Topic:\s*(\S.+)", block)
        # Search for Description instead of Q&A
        d_match = re.search(r"Description:\s*(.+)", block, re.DOTALL)
        
        if d_match:
            topic = topic_match.group(1).strip().lower() if topic_match else "general"
            desc = d_match.group(1).strip()
            
            # Save the description as the answer, and use it for the semantic search index
            dataset_records.append({"topic": topic, "question": "", "answer": desc})
            # Combine the topic and description so the search engine reads both!
            questions_only.append(f"{topic} {desc}")
            
    question_embeddings = embed_model.encode(questions_only, convert_to_tensor=True)
    return dataset_records, question_embeddings, vocab_words

dataset, dataset_embeddings, VOCAB_WORDS = load_and_vectorize_dataset()
print(f"Successfully loaded {len(dataset)} facts.")

# =====================================================================
# TYPO CORRECTION ENGINE
# =====================================================================
def check_for_typos(question: str):
    words = re.findall(r"[a-zA-Z0-9]+", question.lower())
    corrected_words = []
    corrections = []

    for word in words:
        if word in VOCAB_WORDS or len(word) <= 2: 
            corrected_words.append(word)
        else:
            matches = difflib.get_close_matches(word, VOCAB_WORDS, n=1, cutoff=TYPO_CUTOFF)
            if matches:
                corrected_words.append(matches[0])
                corrections.append((word, matches[0]))
            else:
                corrected_words.append(word)

    corrected_question = " ".join(corrected_words)
    return corrected_question, corrections

# =====================================================================
# RETRIEVAL ENGINE
# =====================================================================
def get_relevant_context(user_query, threshold=0.2):
    query_embedding = embed_model.encode(user_query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, dataset_embeddings)[0]
    best_match_idx = torch.argmax(cos_scores).item()
    best_score = cos_scores[best_match_idx].item()
    
    if best_score >= threshold:
        return dataset[best_match_idx] # Returns the whole record dictionary
    return None

# =====================================================================
# MAIN INTERACTIVE LOOP
# =====================================================================
def chat():
    print("\n=========================================")
    print("🤖 ALYai Agent at your service!")
    print("Ask me anything about the TMC, the creator, or about myself. Type 'exit' to quit.")
    print("=========================================\n")
    
    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() == "exit":
            print("Goodbye!")
            break
            
        # Run typo detection
        processed_input = user_input
        corrected_text, corrections = check_for_typos(user_input)
        
        if corrections:
            correction_list = ", ".join(f'"{orig}" -> "{fix}"' for orig, fix in corrections)
            print(f'AI: Did you mean: "{corrected_text}"? ({correction_list})')
            confirm = input("    [Y/n]: ").strip().lower()
            
            if confirm in ("", "y", "yes"):
                processed_input = corrected_text
            else:
                print("    [Proceeding with original text...]")
        
        # --- THE NEW CLARIFICATION RULE ---
        word_count = len(processed_input.split())
        
        if word_count <= 2:
            # If the input is too short, ask the user to be specific
            system_prompt = (
                f"The user just typed a very broad keyword or short phrase: '{processed_input}'. "
                "Respond naturally and politely by asking them what specific information they "
                "would like to know about that topic."
            )
        else:
            # If it's a full question, run the semantic retrieval normally
            matched_record = get_relevant_context(processed_input)
            
            if matched_record:
                topic = matched_record["topic"]
                fact = matched_record["answer"]
            
                # Check if the user specifically asked for a short version
                user_wants_brief = any(word in processed_input.lower() for word in ["brief", "short", "summarize", "summary", "paraphrase"])
            
                if topic in ["vision", "mission", "goal", "philosophy", "slogan"] and not user_wants_brief:
                # Default behavior for official statements: Exact copy
                    system_prompt = (
                    "You are an AI assistant. The user is asking for an official mandate statement. "
                    "You MUST reply by providing the EXACT text from the context below word-for-word. "
                    "Do not paraphrase or reword it.\n\n"
                    f"Context: {fact}"
                )
                else:
                # Behavior for general questions OR when the user asks for a brief version
                    system_prompt = (
                    "You are an AI assistant. Answer the user's question based on the context below. "
                    "Pay very close attention to their instructions. If they ask for a 'brief', 'short', or 'summarized' answer, "
                    "you MUST heavily shorten and paraphrase the context into just one or two simple sentences. "
                    "Otherwise, answer naturally.\n\n"
                    f"Context: {fact}"
                )
            else:
                system_prompt = (
                    "You are Alyssa's AI assistant representing Trinidad Municipal College (TMC). "
                    "If the user asks a question and you don't know the answer, "
                    "simply reply: 'I'm sorry, I don't have that information in my dataset.' "
                    "Do not make up facts or guess."
            )
            
        # Generate the response
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": processed_input}
        ]
        
        text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text_prompt], return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = llm_model.generate(
                **model_inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"AI: {response}\n")

if __name__ == "__main__":
    chat()