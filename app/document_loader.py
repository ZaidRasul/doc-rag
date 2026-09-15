import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader

supported_extensions = [".pdf", ".txt", ".md"]

def load_files(filename: str, file_bytes: bytes) -> List[Document]:

    extension = Path(filename).suffix.lower()
    if extension not in supported_extensions:
        raise ValueError(f"Unsupported file type {extension}")

    if extension == ".pdf":
        return load_pdf(file_name, file_bytes)
    if extension == ".txt":
        return load_txt(file_name, file_bytes)
    if extension == ".md":
        return load_md(file_name, file_bytes)


def load_pdf(file_name: str, file_bytes: bytes) -> List[Document]:
    pass



def load_txt(file_name: str, file_bytes: bytes) -> List[Document]:
    pass



def load_md(file_name: str, file_bytes: bytes) -> List[Document]:
    pass