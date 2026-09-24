
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
    pass


def reciprocal_rank(
        retrieved_ids: list[str],
        relevant_ids: list[str],
) -> float:
    pass