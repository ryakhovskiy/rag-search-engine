import argparse

from lib.hybrid_search import normalize_min_max

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    norm_parser = subparsers.add_parser("normalize", help="Normilizes the BM25 scores to the scale 0..1 using min-max normalization")
    norm_parser.add_argument("scores", type=float, nargs='*')


    rrf_parser = subparsers.add_parser("rrf-search", help="Verifies the model for Semantic Search Embeddings loaded")
    rrf_parser.add_argument("--enhance", type=str, choices=["spell"], help="Query enhancement method",)

    args = parser.parse_args()
    match args.command:
        case "rrf-search":
            print("rrf-search")
        case "normalize":
            normalized = normalize_min_max(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()