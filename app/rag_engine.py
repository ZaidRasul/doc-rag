import uuid
from typing import Any, Dict, List

import chromadb
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class RAGEngine:
    def __init__(self, groq_api_key: str, model_name: str = "openai/gpt-oss-20b", embedding_model_name: str = "all-MiniLM-L6-v2"):
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set. Please set the environment variable.")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name, temperature=0.1, max_tokens=1024)
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name="rag_collection")
        self.document_names: List[str] = []