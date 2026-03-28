def chunk_text(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

from extract_text import extract_text_from_pdf

text = extract_text_from_pdf("resume1.pdf")
chunks = chunk_text(text, chunk_size=200, overlap=50)

print(f"Total chunks: {len(chunks)}")
print("---")
print("First chunk:")
print(chunks[0])
print("---")
print("Second chunk:")
print(chunks[1])