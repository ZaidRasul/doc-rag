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


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is missing from the project-root .env file."
        )

    print("Loading evaluation documents...")

    documents = load_evaluation_documents(
        DATA_DIRECTORIES
    )

    if not documents:
        raise ValueError(
            "No supported documents were found in the data folders."
        )

    print(f"Loaded {len(documents)} document units.")

    # This creates a new in-memory Chroma collection for this run.
    engine = RAGEngine(
        groq_api_key=groq_api_key,
        embedding_model_name="all-MiniLM-L6-v2",
    )

    print("Splitting, embedding, and indexing documents...")

    chunk_count = engine.add_documents(documents)

    print(f"Created {chunk_count} chunks.")

    results = evaluate_retrieval(
        engine=engine,
        dataset_path=str(DATASET_PATH),
        top_k=5,
    )

    save_evaluation_results(
        results=results,
        output_path=str(RESULT_PATH),
    )

    print("\nBaseline results")

    for metric_name, metric_value in results["summary"].items():
        if isinstance(metric_value, float):
            print(f"{metric_name}: {metric_value:.4f}")
        else:
            print(f"{metric_name}: {metric_value}")


if __name__ == "__main__":
    main()