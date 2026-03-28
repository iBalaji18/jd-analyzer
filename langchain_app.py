from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Step 1: Load MULTIPLE PDFs
pdf_files = ["resume1.pdf", "resume2.pdf"]

all_documents = []
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    documents = loader.load()
    all_documents.extend(documents)

print(f"Total pages loaded: {len(all_documents)}")

# Step 2: Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
chunks = splitter.split_documents(all_documents)

print(f"Total chunks: {len(chunks)}")

# Step 3: Embed + Store in Chroma
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")

# Step 4: Build RAG chain
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant analyzing resumes.
Use only the context below to answer the question.
If the answer is not in the context, say "I don't have enough information."

Context: {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

document_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, document_chain)

# Step 5: Chat loop with memory
chat_history = []

print("Chat with your resumes! Type 'exit' to quit.")
print("---")

while True:
    query = input("You: ")
    
    if query.lower() == "exit":
        break
    
    response = chain.invoke({
        "input": query,
        "chat_history": chat_history
    })
    
    answer = response["answer"]
    print(f"Bot: {answer}")
    print("---")
    
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))