import os
import streamlit as st
from dotenv import load_dotenv
from rag_engine import RAGEngine
from document_loader import load_files

load_dotenv()

st.set_page_config(
    page_title="Document RAG",
    page_icon="📄",
    layout="wide",
)

def initialize_session_state():
    pass

def make_file_signature():
    pass