import os
import streamlit as st
from dotenv import load_dotenv
from rag_engine import RAGEngine
from document_loader import load_files

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="Document RAG",
    page_icon="📄",
    layout="wide",
    #write="upload rag docs"
)
st.write("Upload your documents and ask questions about them.")
st.write("Currently supported file types: PDF, TXT, MD, Docx")


def initialize_session_state():
    if "rag_engine" not in st.session_state:
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            st.error(
                "GROQ_API_KEY is missing. Add it to the project-root .env file."
            )
            st.stop()

        with st.spinner("Loading embedding model..."):
            st.session_state.rag_engine = RAGEngine(
                groq_api_key=groq_api_key
            )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "documents_ready" not in st.session_state:
        st.session_state.documents_ready = False

    if "processed_file_signature" not in st.session_state:
        st.session_state.processed_file_signature = None


def make_file_signature():
    pass
