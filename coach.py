import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Streamlit Setup
st.set_page_config(page_title="Knowledge Coach", layout="wide")
st.title("🧠 Knowledge Coach — Smart Document Tutor")
st.sidebar.header("Upload Your Study Materials")

# Get API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("⚠️ OPENAI_API_KEY not found in .env file. Please add it.")
    st.stop()

uploaded_files = st.sidebar.file_uploader(
    "Upload documents (PDF, DOCX, TXT)", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True
)
mode = st.sidebar.selectbox(
    "Select coaching style:", 
    ["Simple Explanation", "Summary", "Deep Dive"]
)

def load_documents(files):
    temp_dir = tempfile.mkdtemp()
    docs = []
    for uploaded_file in files:
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if uploaded_file.name.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif uploaded_file.name.lower().endswith(".docx"):
            loader = Docx2txtLoader(path)
        elif uploaded_file.name.lower().endswith(".txt"):
            loader = TextLoader(path)
        else:
            loader = UnstructuredFileLoader(path)
        docs.extend(loader.load())
    return docs

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_flashcards(content, model):
    flash_prompt = ChatPromptTemplate.from_template(
        "Create 10 concise flashcards (Question–Answer pairs) from the following material:\n\n{content}"
    )
    chain = flash_prompt | model | StrOutputParser()
    return chain.invoke({"content": content})

if uploaded_files:
    with st.spinner("Processing documents..."):
        documents = load_documents(uploaded_files)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, openai_api_key=openai_api_key)

        # Modern LCEL pipeline using RunnablePassthrough
        prompt = ChatPromptTemplate.from_template(
            "You are a knowledgeable tutor. Use the following context to answer:\n\n{context}\n\nQuestion: {question}\n"
        )
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        st.success("✅ Documents loaded successfully!")

        if st.button("Generate Flashcards"):
            with st.spinner("Generating flashcards..."):
                content_text = "\n".join([d.page_content for d in documents[:5]])  # Limit to first 5 docs
                cards = generate_flashcards(content_text, llm)
                st.markdown("### 📚 Generated Flashcards")
                st.write(cards)

        st.markdown("### 💬 Chat With Your Knowledge Coach")
        user_q = st.text_input("Ask your question:")
        if user_q:
            if mode == "Summary":
                user_q = f"Summarize briefly: {user_q}"
            elif mode == "Deep Dive":
                user_q = f"Explain in detailed analytical way: {user_q}"
            else:
                user_q = f"Explain simply: {user_q}"
            with st.spinner("Thinking..."):
                answer = rag_chain.invoke(user_q)
                st.write(answer)
else:
    st.info("📁 Please upload documents to get started.")

st.caption("Built with Streamlit, LangChain (LCEL), and OpenAI · © 2025 Knowledge Coach")