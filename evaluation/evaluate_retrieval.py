import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    hit_rate_at_k,
)

def make_result_id(result: dict[str, Any]) -> str:
   # Identify a result by its source and page number.
    metadata = result.get("metadata", {})
    source = metadata.get("source", "Unknown")
    page = metadata.get("page")

    if page is not None:
        # Page in Chroma is zero-based, while the dataset uses
        # normal one-based page numbers.
        displayed_page = int(page) + 1
        return f"{source}::page_{displayed_page}"

    return source


def evaluate_retrieval(engine, dataset_path: str, top_k: int = 5):
    pass
