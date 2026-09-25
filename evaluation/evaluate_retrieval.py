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
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset file not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not dataset:
        raise ValueError("Evaluation dataset is empty")
    details = []
    

    for example in dataset:
        query = example["query"]
        relevant_ids = example["relevant_ids"]

        retrieved_results = engine.retrieve(query, top_k=top_k)
        retrieved_ids = [
            make_result_id(result) for result in retrieved_results
            ]

        precision = precision_at_k(
            retrieved_ids, relevant_ids, top_k
            )

        recall = recall_at_k(
            retrieved_ids, relevant_ids, top_k
            )

        rr = reciprocal_rank(
            retrieved_ids, relevant_ids
            )

        hit_rate = hit_rate_at_k(
            retrieved_ids, relevant_ids, top_k
            )

        details.append({
            "id": example.get("id"),
            "query": query,
            "relevant_ids": relevant_ids,
            "retrieved_ids": retrieved_ids,
            f"precision_at_{top_k}": precision,
            f"recall_at_{top_k}": recall,
            "reciprocal_rank": rr,
            f"hit_rate_at_{top_k}": hit_rate,


        })

    summary = {
        "number_of_queries": len(details),
        "top_k": top_k,
        f"mean_precision_at_{top_k}": mean(
            result[f"precision_at_{top_k}"]
            for result in details
        ),
        f"mean_recall_at_{top_k}": mean(
            result[f"recall_at_{top_k}"]
            for result in details
        ),
        "mean_reciprocal_rank": mean(
            result["reciprocal_rank"]
            for result in details
        ),
        f"hit_rate_at_{top_k}": mean(
            result[f"hit_rate_at_{top_k}"]
            for result in details
        ),
    }

    return {
        "summary": summary,
        "details": details,
    }

def save_evaluation_results(
    results: dict[str, Any],
    output_path: str,
) -> None:
    """
    Save the evaluation results for later comparisons.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"Results saved to {output_file}")