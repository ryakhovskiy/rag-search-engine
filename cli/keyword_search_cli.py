import argparse

from lib.search_utils import search_from_index
from lib.inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help=f"Search query")

    build_parser = subparsers.add_parser("build", help="Build an inverted index and store it to disk")

    load_parser = subparsers.add_parser("load", help="Loads an inverted index from disk")

    tf_parser = subparsers.add_parser("tf", help="Search term frequency, how ofthen a term appears in a movie description")
    tf_parser.add_argument("doc_id", type=int, help=f"Document ID")
    tf_parser.add_argument("term", type=str, help=f"Search term")

    idf_parser = subparsers.add_parser("idf", help="Calculate inverse document frequency index")
    idf_parser.add_argument("term", type=str, help=f"Search term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate term frequency * inverse document frequency for document ranking")
    tfidf_parser.add_argument("doc_id", type=int, help=f"Document ID")
    tfidf_parser.add_argument("term", type=str, help=f"Search term")

    search_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    search_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=InvertedIndex.BM25_K1, help="Tunable BM25 K1 parameter")

    args = parser.parse_args()
    invIndex = InvertedIndex()

    match args.command:
        case "search":
            invIndex.load()
            print(f"Searching for: {args.query}")
            results = search_from_index(args.query, invIndex.index, invIndex.docmap)
            for res in results:
                print(res)
        case "build":
            invIndex.build()
            invIndex.save()
        case "load":
            invIndex.load()
        case "tf":
            invIndex.load()
            count = invIndex.get_tf(args.doc_id, args.term)
            print(f"The term {args.term} in the document with ID {args.doc_id} appears {count} times")
        case "idf":
            invIndex.load()
            idf = invIndex.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            invIndex.load()
            tf_idf = invIndex.get_tfidf(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            invIndex.load()
            bm25idf = invIndex.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            invIndex.load()
            bm25tf = invIndex.get_bm25_tf(doc_id=args.doc_id, term=args.term, k1=args.k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()