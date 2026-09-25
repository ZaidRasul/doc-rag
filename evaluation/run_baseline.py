import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document

from app.document_loader import (
    load_docx,
    load_md,
    load_pdf,
    load_txt,
)
from app.rag_engine import RAGEngine
from evaluation.evaluate_retrieval import (
    evaluate_retrieval,
    save_evaluation_results,
)