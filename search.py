from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from extract_text import extract_text_from_pdf
from chunk_text import chunk_text
from embed import embed_chunks

def search(query, chunks, embeddings, top_k=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings)
    
    top_indices = scores[0].argsort()[::-1][:top_k]
    
    results = [chunks[i] for i in top_indices]
    return results

text = extract_text_from_pdf("resume1.pdf")
chunks = chunk_text(text, chunk_size=500, overlap=150)
embeddings = embed_chunks(chunks)

query = "What are this person's skills?"
results = search(query, chunks, embeddings)

print("Top matching chunks:")
print("---")
for i, result in enumerate(results):
    print(f"Match {i+1}:")
    print(result)
    print("---")