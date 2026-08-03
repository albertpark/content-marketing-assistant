"""Sample content-request dataset used to evaluate the ContentAlchemy pipeline
end-to-end in LangSmith. Starter set — add more examples as real user queries
surface real failure modes worth regression-testing."""

from __future__ import annotations

from langsmith import Client

DATASET_NAME = "contentalchemy-content-requests"

_EXAMPLE_QUERIES: list[str] = [
    "Write a blog post about the benefits of remote work for software teams",
    "Write a blog post introducing our new AI-powered analytics dashboard",
    "Write a blog post about sustainable packaging trends in e-commerce",
]


def sync_dataset(client: Client | None = None) -> str:
    """Creates the dataset if it doesn't exist yet, and adds any of
    _EXAMPLE_QUERIES not already present. Safe to call on every eval run."""
    client = client or Client()

    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=(
                "Sample content-marketing requests for evaluating the "
                "ContentAlchemy pipeline end-to-end."
            ),
        )

    existing = {ex.inputs.get("user_query") for ex in client.list_examples(dataset_id=dataset.id)}
    missing = [q for q in _EXAMPLE_QUERIES if q not in existing]
    if missing:
        client.create_examples(
            dataset_id=dataset.id,
            examples=[{"inputs": {"user_query": q}} for q in missing],
        )
    return DATASET_NAME
