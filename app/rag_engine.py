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
        self.collection = self.chroma_client.create_collection(name="documents", metadata={"hnsw:space": "cosine"})
        self.document_names: List[str] = []

    def query(self, query: str, top_k: int = 3) -> str:
        pass

    def doc_splitter(self, documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 200) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
                                                  )
        split_docs = splitter.split_documents(documents)
        return split_docs

    def embedder(self, documents: List[Document]) -> List[List[float]]:
        pass

    def add_documents(self, documents: List[Document]) -> int:
        pass
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        pass

    def answer(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        pass

