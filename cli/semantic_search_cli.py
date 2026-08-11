import argparse

from lib.semantic_search import verify_model
from lib.semantic_search import embed_text

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_parser = subparsers.add_parser("verify", help="Verifies the model for Semantic Search Embeddings loaded")

    embeddings_parser = subparsers.add_parser("embed_text", help="Verifies the model for Semantic Search Embeddings loaded")
    embeddings_parser.add_argument("text", type=str, help="Text to generate embeddings")

    args = parser.parse_args()
    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()