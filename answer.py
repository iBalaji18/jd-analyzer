from groq import Groq
from dotenv import load_dotenv
import os
from extract_text import extract_text_from_pdf
from chunk_text import chunk_text
from embed import embed_chunks
from search import search

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

def answer_question(query, chunks, embeddings):
    relevant_chunks = search(query, chunks, embeddings)
    
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"""You are a helpful assistant analyzing a resume.

Use only the context below to answer the question.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

text = extract_text_from_pdf("resume1.pdf")
chunks = chunk_text(text, chunk_size=500, overlap=150)
embeddings = embed_chunks(chunks)

query = "What programming languages does this person know?"
answer = answer_question(query, chunks, embeddings)

print("Question:", query)
print("---")
print("Answer:", answer)