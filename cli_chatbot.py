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
import uuid

load_dotenv()

# Load and process PDF
pdf_path = input("Enter PDF path: ")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    collection_name=f"session_{uuid.uuid4().hex}"
)

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Two modes — Q&A and Summarization
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant answering questions about a resume.
Use only the context below to answer.
If the answer is not in context, say "I don't have enough information."

Context: {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant summarizing resumes.
When asked to summarize, provide a structured summary including:
- Candidate name and contact
- Education
- Skills
- Experience
- Projects

Use only the context below.

Context: {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

def get_chain(mode):
    if mode == "2":
        document_chain = create_stuff_documents_chain(llm, summarize_prompt)
    else:
        document_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(retriever, document_chain)

# CLI interface
print("\n✅ Resume loaded successfully!")
print("\nSelect mode:")
print("1. Q&A Mode — ask specific questions")
print("2. Summarization Mode — summarize the resume")

mode = input("\nEnter mode (1 or 2): ")
chain = get_chain(mode)
chat_history = []

print("\nChatbot ready! Type 'switch' to change mode, 'exit' to quit.")
print("---")

while True:
    query = input("You: ")
    
    if query.lower() == "exit":
        print("Goodbye!")
        break
    
    if query.lower() == "switch":
        mode = "2" if mode == "1" else "1"
        chain = get_chain(mode)
        current = "Summarization" if mode == "2" else "Q&A"
        print(f"Switched to {current} mode!")
        continue
    
    response = chain.invoke({
        "input": query,
        "chat_history": chat_history
    })
    
    answer = response["answer"]
    print(f"Bot: {answer}")
    print("---")
    
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))