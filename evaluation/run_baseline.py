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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORIES = [
    PROJECT_ROOT / "data" / "pdf",
    PROJECT_ROOT / "data" / "text-files",
]
DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "dataset"
    / "retrieval_dataset.json"
)
RESULT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "baseline_v1.json"
)


def load_evaluation_documents(
    data_directories: list[Path],
) -> list[Document]:
    """
    Load supported source documents from the existing data folders.
    """
    all_documents = []

    loaders = {
        ".pdf": load_pdf,
        ".txt": load_txt,
        ".md": load_md,
        ".docx": load_docx,
    }

    for directory in data_directories:
        if not directory.exists():
            print(f"Skipping missing directory: {directory}")
            continue

        for file_path in sorted(directory.rglob("*")):
            if not file_path.is_file():
                continue

            extension = file_path.suffix.lower()
            loader = loaders.get(extension)

            if loader is None:
                print(f"Skipping unsupported file: {file_path}")
                continue

            print(f"Loading: {file_path.name}")

            file_documents = loader(
                file_name=file_path.name,
                file_bytes=file_path.read_bytes(),
            )

            all_documents.extend(file_documents)

    return all_documents
