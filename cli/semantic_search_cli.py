import argparse

from lib.semantic_search import verify_model
from lib.semantic_search import embed_text
from lib.semantic_search import verify_embeddings
from lib.semantic_search import embed_query_text
from lib.semantic_search import search
from lib.semantic_search import semantic_chunk
from lib.search_utils import chunk
from lib.chunked_semantic_search import embed_chunks
from lib.chunked_semantic_search import search_chunks

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_parser = subparsers.add_parser("verify", help="Verifies the model for Semantic Search Embeddings loaded")

    embeddings_parser = subparsers.add_parser("embed_text", help="Verifies the model for Semantic Search Embeddings loaded")
    embeddings_parser.add_argument("text", type=str, help="Text to generate embeddings")

    verify_parser = subparsers.add_parser("verify_embeddings", help="Verifies the embeddings properly loaded")

    embed_qt = subparsers.add_parser("embed_query", help="Generates embeddings for query text")
    embed_qt.add_argument("query", type=str, help="Query text to generate embeddings")

    search_parser = subparsers.add_parser("search", help="Search using semantic similarity")
    search_parser.add_argument("query", type=str, help="Query for the semantic search")
    search_parser.add_argument("--limit", type=int, default=5, help="Limit resultset")

    chunk_parser = subparsers.add_parser("chunk", help="Chunks document")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Fixed chunk size")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="chunk overlap with the previuos chunk to preserve context")

    schunk_parser = subparsers.add_parser("semantic_chunk", help="Chunks document semantically")
    schunk_parser.add_argument("text", type=str, help="Text to chunk")
    schunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Max fixed chunk size")
    schunk_parser.add_argument("--overlap", type=int, default=0, help="chunk overlap with the previuos chunk to preserve context")

    ec_parser = subparsers.add_parser("embed_chunks", help="Load or create chunked semantic embeddings")

    search_chunked = subparsers.add_parser("search_chunked", help="Search using chunks of documents semantically")
    search_chunked.add_argument("query", type=str, help="Query for the semantic search")
    search_chunked.add_argument("--limit", type=int, default=5, help="Limit resultset")

    args = parser.parse_args()
    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings() 
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case "chunk":
            print(f"Chunking {len(args.text)} characters")
            res = chunk(args.text, args.chunk_size, args.overlap)
            for i in range(0, len(res)):
                print(f"{i+1}. {res[i]}")
        case "semantic_chunk":
            print(f"Semantically chunking {len(args.text)} characters")
            res = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            for i in range(0, len(res)):
                print(f"{i+1}. {res[i]}")
        case "embed_chunks":
            embed_chunks()
        case "search_chunked":
            search_chunks(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()