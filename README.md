# 📄 JD Analyzer — AI-Powered Resume & Job Description Matcher

A RAG-based application that analyzes your resume against a job description and identifies skill gaps using local embeddings and an LLM.

## 🚀 Demo
Upload your resume (PDF) + paste a job description → Get instant analysis of your skills and gaps.

## 🛠️ Tech Stack
- **Python** — Core language
- **Sentence Transformers (MiniLM)** — Local embeddings, no API cost
- **Cosine Similarity** — Semantic search
- **Groq (Llama 3.3)** — LLM for analysis
- **Streamlit** — Web UI
- **PyPDF** — PDF text extraction

## 🧠 How It Works
1. Resume PDF is loaded and split into chunks
2. Each chunk is converted into a vector using MiniLM embeddings
3. User's query is embedded and compared against chunks using cosine similarity
4. Top matching chunks are retrieved and sent to Groq LLM with the job description
5. LLM reasons about skill matches and gaps

## 📦 Installation
```bash
git clone https://github.com/yourusername/jd-analyzer.git
cd jd-analyzer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## ⚙️ Setup
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key
```

## ▶️ Run
```bash
streamlit run app.py
```

## 🔍 Features
- Upload any resume in PDF format
- Paste any job description
- Get key skills analysis
- Get work experience summary
- Get missing skills identification

## 📌 Note
Built manually without LangChain abstraction to understand RAG fundamentals from scratch.