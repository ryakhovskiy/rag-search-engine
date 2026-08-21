import argparse

from lib.hybrid_search import normalize_min_max
from lib.hybrid_search import weighted_search
from lib.hybrid_search import rrf_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    norm_parser = subparsers.add_parser("normalize", help="Normilizes the BM25 scores to the scale 0..1 using min-max normalization")
    norm_parser.add_argument("scores", type=float, nargs='*')

    ws_parser = subparsers.add_parser("weighted-search", help="Weighted search using Semantic Search and BM25, normalized score")
    ws_parser.add_argument("query", type=str, help="Terms to search for")
    ws_parser.add_argument("--alpha", type=float, default=0.5, help="Alpha constant to control the weighting between the two scores")
    ws_parser.add_argument("--limit", type=int, default=5, help="Limit the resultset")

    rrf_parser = subparsers.add_parser("rrf-search", help="Verifies the model for Semantic Search Embeddings loaded")
    rrf_parser.add_argument("query", type=str, help="Terms to search for")
    rrf_parser.add_argument("-k", type=int, default=60, help="controls how much more weight to give to higher-ranked results vs. lower-ranked ones")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Limit the resultset")

    args = parser.parse_args()
    match args.command:
        case "normalize":
            normalized = normalize_min_max(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search(args.query, args.k, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()