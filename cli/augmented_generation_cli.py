import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Summarizes search results")
    summarize_parser.add_argument("query", type=str, help="Search query for RRF-Search")

    args = parser.parse_args()
    match args.command:
        case "rag":
            query = args.query
            from lib.rag_pipeline import run_rag_pipeline
            run_rag_pipeline(query)
        case "summarize":
            query = args.query
            from lib.rag_pipeline import run_summarization
            run_summarization(query)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()