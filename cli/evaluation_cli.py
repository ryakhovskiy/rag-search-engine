import argparse
import json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )
    args = parser.parse_args()
    limit = args.limit

    test_cases = load_golden_dataset()
    from lib.hybrid_search import rrf_search
    print(f"k={limit}")
    for test_case in test_cases:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]
        retrieved_docs = rrf_search(query, k=60, limit=limit)
        retrieved_titles = {x["title"] for x in retrieved_docs}
        relevant_retrieved = len(retrieved_titles & set(relevant_docs))
        precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0.0
        recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0.0
        f1 = 2 * (precision * recall) / (precision + recall)
        print("")
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {retrieved_titles}")
        print(f"  - Relevant: {relevant_docs}")


def load_golden_dataset() -> dict:
    file_path = Path("data/golden_dataset.json")
    if not file_path.is_file():
        print(f"file not found: {file_path}")
        return dict()
    with open(file_path, "r") as f:
        data = json.load(f)
        golden_dataset = data["test_cases"]
        print(f"golden dataset loaded successfully, {len(golden_dataset)} records")
        return golden_dataset


    

if __name__ == "__main__":
    main()