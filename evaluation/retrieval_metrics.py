
def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """
    Fraction of the first k retrieved items that are relevant.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero")

    retrieved_at_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)

    relevant_retrieved = sum(
        retrieved_id in relevant_set
        for retrieved_id in retrieved_at_k
    )

    return relevant_retrieved / k



def recall_at_k(
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int,
) -> float:
    relevant_set = set(relevant_ids)

    if not relevant_set:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    relevant_retrieved = len(retrieved_at_k & relevant_set)

    return relevant_retrieved / len(relevant_set)


def reciprocal_rank(
        retrieved_ids: list[str],
        relevant_ids: list[str],
) -> float:
    relevant_set = set(relevant_ids)

    for rank, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in relevant_set:
            return 1.0 / rank

    return 0.0


def hit_rate_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    relevant_set = set(relevant_ids)
    return float(
        any(
            retrieved_id in relevant_set
            for retrieved_id in retrieved_ids[:k]
        )
    )