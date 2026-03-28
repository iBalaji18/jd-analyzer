from sentence_transformers import SentenceTransformer
from extract_text import extract_text_from_pdf
from chunk_text import chunk_text

def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return embeddings

text = extract_text_from_pdf("resume1.pdf")
chunks = chunk_text(text, chunk_size=200, overlap=50)
embeddings = embed_chunks(chunks)

print(f"Total chunks embedded: {len(embeddings)}")
print(f"Each embedding has {len(embeddings[0])} numbers")
print("---")
print("First embedding (first 10 numbers):")
print(embeddings[0][:10])