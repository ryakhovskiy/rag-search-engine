import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Summarizes search results")
    summarize_parser.add_argument("query", type=str, help="Search query for RRF-Search")

    citation_parser = subparsers.add_parser("citations", help="Adds citations to the search results")
    citation_parser.add_argument("query", type=str, help="Search query for RRF-Search")
    citation_parser.add_argument("--limit", type=int, default=5, help="Limit the resultset")

    question_parser = subparsers.add_parser("question", help="Run a question against the movies data")
    question_parser.add_argument("question", type=str, help="User question")
    question_parser.add_argument("--limit", type=int, default=5, help="Limit the resultset")

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
        case "citations":
            query = args.query
            from lib.rag_pipeline import run_citations
            run_citations(query=query, limit=args.limit)
        case "question":
            question = args.question
            from lib.rag_pipeline import run_question
            run_question(question, limit=args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()