import streamlit as st
from extract_text import extract_text_from_pdf
from chunk_text import chunk_text
from embed import embed_chunks
from search import search
from groq import Groq
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_question(query, chunks, embeddings):
    relevant_chunks = search(query, chunks, embeddings)
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"""You are an expert HR assistant analyzing a resume against a job description.

Use only the context below to answer the question.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# UI starts here
st.title("📄 JD Analyzer")
st.subheader("Match your resume against a job description")

resume_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")
jd_text = st.text_area("Paste the Job Description here", height=200)

if st.button("Analyze"):
    if resume_file and jd_text:
        with st.spinner("Analyzing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(resume_file.read())
                tmp_path = tmp.name
            
            text = extract_text_from_pdf(tmp_path)
            chunks = chunk_text(text, chunk_size=500, overlap=150)
            embeddings = embed_chunks(chunks)
            
            st.subheader("Results")
            
            questions = [
                "What are this person's key skills?",
                "What is this person's work experience?",
                "Based on the resume, what skills might be missing for this job description: " + jd_text[:500]
            ]
            
            for question in questions:
                with st.expander(question):
                    answer = answer_question(question, chunks, embeddings)
                    st.write(answer)
    else:
        st.warning("Please upload a resume and paste a job description!")