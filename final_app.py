import streamlit as st
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
import tempfile
import os
import uuid

load_dotenv()

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.chain = None
    st.session_state.chat_history = []
    st.session_state.messages = []

st.set_page_config(page_title="JD Analyzer", page_icon="📄", layout="wide")
st.title("📄 JD Analyzer")
st.subheader("Upload resumes and chat with them!")

# Session state — remembers things between interactions
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for uploading
with st.sidebar:
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF resumes",
        type="pdf",
        accept_multiple_files=True
    )
    
    jd_text = st.text_area("Paste Job Description (optional)", height=150)
    
    if st.button("Process Resumes") and uploaded_files:
        with st.spinner("Processing resumes..."):
            
            # Save uploaded files temporarily
            all_documents = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                all_documents.extend(documents)
            
            # Chunk
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
            chunks = splitter.split_documents(all_documents)
            
            # Embed + Store
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(
            chunks,
            embeddings,
            collection_name=f"session_{uuid.uuid4().hex}")
            
            # Build chain
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=os.getenv("GROQ_API_KEY")
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert HR assistant analyzing resumes.
Each document in the context has metadata showing which file it came from.
Always mention the candidate's name when referring to their skills or experience.
Use only the context below to answer questions.
If the answer is not in the context, say "I don't have enough information."

Context: {context}

""" + (f"Job Description: {jd_text}" if jd_text else "")),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])
            
            document_chain = create_stuff_documents_chain(llm, prompt)
            st.session_state.chain = create_retrieval_chain(retriever, document_chain)
            st.session_state.chat_history = []
            st.session_state.messages = []
            
        st.success(f"✅ {len(uploaded_files)} resume(s) processed!")

# Chat interface
if st.session_state.chain:
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    query = st.chat_input("Ask anything about the resumes...")
    
    if query:
        # Show user message
        with st.chat_message("user"):
            st.write(query)
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chain.invoke({
                    "input": query,
                    "chat_history": st.session_state.chat_history
                })
                answer = response["answer"]
                st.write(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append(HumanMessage(content=query))
        st.session_state.chat_history.append(AIMessage(content=answer))

else:
    st.info("👈 Upload resumes from the sidebar to get started!")