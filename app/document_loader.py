import os
import tempfile
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader,TextLoader,UnstructuredMarkdownLoader

supported_extensions = [".pdf", ".txt", ".md"]

def load_files(uploaded_files) -> List[Document]:
    all_docs = []
    # Load multiple files if needed
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_bytes = uploaded_file.getvalue()
        extension = Path(file_name).suffix.lower()
        # check if the file extension is supported
        if extension not in supported_extensions:
            raise ValueError(f"Unsupported file type {extension}")
        # check extensions and call the appropriate loader function
        if extension == ".pdf":
            documents = load_pdf(file_name, file_bytes)
        if extension == ".txt":
            documents = load_txt(file_name, file_bytes)
        if extension == ".md":
            documents = load_md(file_name, file_bytes)

        all_docs.extend(documents)

    return all_docs

def load_pdf(file_name: str, file_bytes: bytes) -> List[Document]:
    all_docs = []
    path = Path(file_name)
    



def load_txt(file_name: str, file_bytes: bytes) -> List[Document]:
    pass



def load_md(file_name: str, file_bytes: bytes) -> List[Document]:
    pass