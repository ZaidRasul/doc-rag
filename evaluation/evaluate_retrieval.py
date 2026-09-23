import json

from evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

def evaluate_retrieval(engine, dataset_path: str, top_k: int = 5):
    pass
