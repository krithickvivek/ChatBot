# Intelligent Enterprise Chatbot

## Description
AI-powered enterprise chatbot using NLP to answer employee queries on HR, IT, and company policies. It processes documents, extracts summaries and keywords, supports multi-user interactions, personalized replies with emojis, and 2FA for secure, fast, and scalable access.

## Features
- **Document Processing:** Upload PDFs or text files to extract relevant information.  
- **Query Handling:** Answers questions related to HR, IT support, company events, and policies.  
- **Personalized Interaction:** Uses your name, emojis, and friendly responses for greetings and thanks.  
- **Multi-user Support:** Handles multiple users concurrently with optimized response times.  
- **Security:** Two-Factor Authentication (2FA) via email for secure access.  
- **Summary & Keyword Extraction:** Uses pre-trained models for fast and accurate responses.

## Technologies Used
- Python  
- Hugging Face Transformers  
- Sentence Transformers  
- PyPDF2  
- NLTK  
- PyTorch / Torch  
- Google Colab (for demonstration)

## Goal
To provide a secure, intelligent, and scalable virtual assistant that helps employees efficiently navigate organizational policies and procedures.

## Usage
1. Upload a PDF or text document.  
2. Ask questions related to the document or organizational policies.  
3. Receive summarized answers or relevant sentences with personalized emojis.  
4. Exit the chatbot.

## Program Code 
### Imports
```python
!pip install transformers sentence-transformers PyPDF2 nltk --quiet
     

import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from google.colab import files
     

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

```
### Load Models
```python
print("Loading models... please wait.")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Models loaded successfully!\n")
```
### Upload and Extract
```python
print("Upload a document (PDF)...")
uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]

def extract_text_from_pdf(path):
    reader = PyPDF2.PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

document_text = extract_text_from_pdf(pdf_path)
print("\n✅ Document uploaded and extracted successfully!\n")
```
### Summarize the Document
```python
print("Generating summary... please wait.\n")
chunk = document_text[:3000] if len(document_text) > 3000 else document_text
summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
print("📄 Summary:\n")
print(summary[0]['summary_text'])
```
### Keywords Extraction
```python
def extract_keywords(text, num_keywords=10):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in tokens if w not in stop_words and w not in string.punctuation]
    freq = {}
    for word in filtered:
        freq[word] = freq.get(word, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:num_keywords]]

keywords = extract_keywords(document_text)
print("\n🔑 Keywords:", ", ".join(keywords))
```
### Simple Chatbot Loop (Document Q/A)
```python
import random

user_name = input("Enter your name 😊 :")

emojis = ["🙂", "😊", "👍", "🙌", "💡", "🤓", "✨", "😄", "🎯", "📝"]
thank_emojis = ["🙏", "💖", "😊", "🌟", "😄"]

def chatbot(query):
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    greeting_responses = ["Hello", "Hi there", "Hey", "Greetings"]

    thanks = ["thanks", "thank you", "good job", "well done", "appreciate"]
    thanks_responses = [
        f"You're welcome {user_name}! {random.choice(thank_emojis)}",
        f"My pleasure, {user_name}! {random.choice(thank_emojis)}",
        f"Happy to help you {user_name}! {random.choice(thank_emojis)}"
    ]

    if any(word in query.lower() for word in greetings):
        return f"{random.choice(greeting_responses)} {user_name}! 👋 How can I help you today? {random.choice(emojis)}"

    if any(word in query.lower() for word in thanks):
        return random.choice(thanks_responses)

    if sentence_embeddings.nelement() == 0:
        return f"Sorry, I couldn't process the document properly. No content found. 😔"

    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]

    top_result = torch.topk(cos_scores, k=1)
    best_sentences = [sentences[idx].strip() for idx in top_result[1]]
    answer = " ".join(best_sentences)

    return f"{answer} {random.choice(emojis)}"

while True:
  q = input(f"\n{user_name}: ")
  if q.lower() in ["exit", "quit", "bye bot", "bye"]:
        print(f"Chatbot: Goodbye {user_name}! 👋 Have a great day! {random.choice(emojis)}")
        break
  ans = chatbot(q)
  print(f"Chatbot: {ans}")
```

## Output 
<img width="746" height="493" alt="image" src="https://github.com/user-attachments/assets/c27af921-2339-4390-8374-c8b9c8d69191" />

## Conculsion 
In conclusion, the Intelligent Enterprise Chatbot provides a smart, secure, and user-friendly solution for automating employee queries, improving communication, and enhancing organizational efficiency through intelligent document understanding and personalized interaction.

