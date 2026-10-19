# RAG Search Engine

My personal RAG practicing.

A Python project for searching a movie dataset with keyword matching, BM25, semantic embeddings, hybrid retrieval, retrieval augmented generation (RAG), and images.

## Setup

Requires Python **3.14 or newer**. 

Run commands from the repository root so that `data/` and `cache/` resolve correctly.

Install dependencies with uv:

```bash
uv sync
mkdir -p data cache
```

The `data/` directory is ignored by Git and must be supplied locally:

| File | Required by | Format |
| --- | --- | --- |
| `data/movies.json` | Indexing, movie retrieval, and image embedding verification | A JSON object with a `movies` array; each movie has an integer `id`, string `title`, and string `description`. |
| `data/stopwords.txt` | Keyword and hybrid retrieval | Plain text stopwords, one per line. |
| `data/golden_dataset.json` | Evaluation only | A JSON object with a `test_cases` array; each case has a `query` and a `relevant_docs` array of movie titles. |

Example movie dataset (can be transformed from https://github.com/erik-sytnyk/movies-list/blob/master/db.json):

```json
{
  "movies": [
    {
      "id": 1,
      "title": "Example Space Adventure",
      "description": "An astronaut journeys across the galaxy."
    }
  ]
}
```

Example evaluation dataset; titles must match the movie dataset:

```json
{
  "test_cases": [
    {
      "query": "space adventure",
      "relevant_docs": ["Example Space Adventure"]
    }
  ]
}
```

For LLM features, set the following in a `.env` file in the repository root, or export the environment variable:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
```

The LLM client uses OpenRouter at `https://openrouter.ai/api/v1` with model `openrouter/free`. The key is required for query enhancement, LLM reranking, LLM evaluation, RAG, image query rewriting, and the LLM smoke test. The hybrid CLI imports the LLM client at startup, so **every hybrid CLI command requires the key**, including `normalize` and `--help`.

Embedding commands use `all-MiniLM-L6-v2`; image commands use `clip-ViT-B-32`. Cross encoder reranking uses `cross-encoder/ms-marco-TinyBERT-L2-v2` on CPU. These models may download on first use. Index and embedding files are saved under `cache/`; create that directory before building them.

## Running commands

Use `uv run python -m cli.<module>` from the repository root. Quote queries or text containing spaces.

```bash
uv run python -m cli.keyword_search_cli build
uv run python -m cli.keyword_search_cli bm25search "space adventure"
uv run python -m cli.semantic_search_cli search "a journey through space" --limit 5
uv run python -m cli.hybrid_search_cli rrf-search "space adventure" --limit 5
uv run python -m cli.augmented_generation_cli question "Which movies feature space travel?"
```

Keyword search also has an installed console entry point:

```bash
uv run keyword-search bm25search "space adventure"
```

All argparse CLIs accept `--help`. For CLIs with subcommands, append `--help` to a subcommand to see its arguments:

```bash
uv run python -m cli.semantic_search_cli --help
uv run python -m cli.hybrid_search_cli rrf-search --help
```

In the references below, angle brackets indicate required positional arguments and square brackets indicate optional arguments. Do not type the brackets.

## Keyword search

Prefix: `uv run python -m cli.keyword_search_cli`

Build the inverted index before using the other keyword commands. It indexes movie titles and descriptions, applying tokenization, stopword removal, and stemming.

| Command | Purpose |
| --- | --- |
| `build` | Build the inverted index from the movie dataset and save it to disk. |
| `load` | Load the saved index and report its status. |
| `search <query>` | Look up movie IDs and titles using keyword matches; see the implementation note below. |
| `tf <doc_id> <term>` | Print the term frequency in a movie's indexed title and description. |
| `idf <term>` | Print inverse document frequency. |
| `tfidf <doc_id> <term>` | Print the TF-IDF score for a movie and term. |
| `bm25idf <term>` | Print BM25 inverse document frequency. |
| `bm25tf <doc_id> <term> [k1] [b]` | Print the BM25 term-frequency component; defaults: `k1=1.5`, `b=0.75`. These are positional arguments. |
| `bm25search <query>` | Retrieve up to five movies ranked by BM25 and print their IDs, titles, and scores. |

Use a movie's actual `id` from `data/movies.json` for `doc_id`.

```bash
uv run python -m cli.keyword_search_cli build
uv run python -m cli.keyword_search_cli load
uv run python -m cli.keyword_search_cli search "space adventure"
uv run python -m cli.keyword_search_cli tf 1 astronaut
uv run python -m cli.keyword_search_cli idf astronaut
uv run python -m cli.keyword_search_cli tfidf 1 astronaut
uv run python -m cli.keyword_search_cli bm25idf astronaut
uv run python -m cli.keyword_search_cli bm25tf 1 astronaut 1.5 0.75
uv run python -m cli.keyword_search_cli bm25search "space adventure"
```

Current implementation notes: `search` calls `tokenize_text` with an extra argument and raises `TypeError`; use `bm25search` for retrieval. The `bm25tf` CLI accepts `b` but does not forward it to the scoring function, so the effective value remains `0.75`.

## Semantic search and chunking

Prefix: `uv run python -m cli.semantic_search_cli`

Movie search loads cached embeddings or creates them when needed.

| Command | Purpose and options |
| --- | --- |
| `verify` | Load the embedding model and print its details and maximum sequence length. |
| `embed_text <text>` | Print the first three embedding dimensions and vector size for text. |
| `verify_embeddings` | Load or create movie embeddings and print their count and shape. |
| `embed_query <query>` | Embed a query and print its first three dimensions and shape. |
| `search <query> [--limit N]` | Rank movies by cosine similarity; default limit: `5`. |
| `chunk <text> [--chunk-size N] [--overlap N]` | Print fixed-size chunks; defaults: size `200`, overlap `0`. See the implementation note below. |
| `semantic_chunk <text> [--max-chunk-size N] [--overlap N]` | Split text into sentence groups; defaults: `4` sentences per chunk, `0` overlapping sentences. |
| `embed_chunks` | Load or create embeddings for movie description chunks. |
| `search_chunked <query> [--limit N]` | Rank movies using each movie's best matching chunk; default limit: `5`. |

```bash
uv run python -m cli.semantic_search_cli verify
uv run python -m cli.semantic_search_cli embed_text "A space adventure"
uv run python -m cli.semantic_search_cli verify_embeddings
uv run python -m cli.semantic_search_cli embed_query "space adventure"
uv run python -m cli.semantic_search_cli search "an astronaut exploring a distant planet" --limit 3
uv run python -m cli.semantic_search_cli chunk "A long movie description to split." --chunk-size 20 --overlap 5
uv run python -m cli.semantic_search_cli semantic_chunk "An astronaut leaves Earth. The ship lands. A mystery begins." --max-chunk-size 2 --overlap 1
uv run python -m cli.semantic_search_cli embed_chunks
uv run python -m cli.semantic_search_cli search_chunked "an astronaut exploring a distant planet" --limit 3
```

For positive chunk sizes, keep overlap nonnegative and smaller than the chunk size. A size of `0` returns the entire input as one chunk.

Current implementation note: `chunk` passes a string to a helper expecting a word list, so it currently chunks individual non-space characters. `semantic_chunk` groups sentences split at punctuation; it does not use embeddings to choose boundaries. Cached movie description chunks use four sentences with one sentence of overlap, independently of the standalone chunking flags.

## Hybrid search

Prefix: `uv run python -m cli.hybrid_search_cli`

Combines BM25 with chunked semantic retrieval. Missing indexes and chunk embeddings are built automatically; local data files and the `cache/` directory are still required.

| Command | Purpose |
| --- | --- |
| `normalize [scores ...]` | Print min-max normalized floating-point scores on a `0..1` scale. Equal scores normalize to `1.0`; no scores produce no output. |
| `weighted-search <query> [--alpha A] [--limit N]` | Request a weighted blend of normalized BM25 and semantic scores; defaults: `alpha=0.5`, limit `5`. See the implementation note below. |
| `rrf-search <query> [options]` | Combine ranked results with reciprocal rank fusion (RRF) and print score tables. |

Options for `rrf-search`:

| Option | Default | Meaning |
| --- | --- | --- |
| `-k N` | `60` | RRF rank constant in `1 / (k + rank)`; smaller values favor higher ranks more strongly. The flag is `-k`, not `--k`. |
| `--limit N` | `5` | Number of returned results. |
| `--enhance spell\|rewrite\|expand` | Disabled | Use the LLM to correct spelling, rewrite the query, or generate related terms. The returned text replaces the query for all three modes. |
| `--rerank-method individual\|batch\|cross_encoder` | Disabled | Rerank five times the requested result count, then retain the requested limit. `individual` scores candidates with separate LLM calls; `batch` requests one LLM ranking; `cross_encoder` uses a local model. |
| `--evaluate` | Disabled | Use an LLM to rate each result from `0` (not relevant) to `3` (highly relevant). |

The weighted blend formula is `alpha * BM25 + (1 - alpha) * semantic`; use alpha values between `0` and `1`.

```bash
uv run python -m cli.hybrid_search_cli normalize 1 2 3
uv run python -m cli.hybrid_search_cli weighted-search "space adventure" --alpha 0.7 --limit 5
uv run python -m cli.hybrid_search_cli rrf-search "space adventure" -k 60 --limit 5
uv run python -m cli.hybrid_search_cli rrf-search "spcae adventure" --enhance spell
uv run python -m cli.hybrid_search_cli rrf-search "a movie about exploring space" --enhance rewrite
uv run python -m cli.hybrid_search_cli rrf-search "space adventure" --enhance expand
uv run python -m cli.hybrid_search_cli rrf-search "space adventure" --rerank-method individual --limit 3
uv run python -m cli.hybrid_search_cli rrf-search "space adventure" --rerank-method cross_encoder --limit 3
uv run python -m cli.hybrid_search_cli rrf-search "space adventure" --evaluate
```

Current implementation notes: `weighted-search` can raise `KeyError` for missing semantic scores and can use BM25 alone instead of the blend. The accepted `batch` reranking mode currently fails because its LLM helper does not return the parsed ranking.

## Retrieval augmented generation

Prefix: `uv run python -m cli.augmented_generation_cli`

These commands retrieve movies with RRF, list the retrieved titles, and generate a response through OpenRouter.

| Command | Purpose |
| --- | --- |
| `rag <query>` | Generate a bullet list answer using the top five retrieved movies. |
| `summarize <query>` | Synthesize the top five results into a short summary. |
| `citations <query> [--limit N]` | Generate an answer with numbered source citations; default limit: `5`. |
| `question <question> [--limit N]` | Answer a question using the retrieved movie context; default limit: `5`. |

```bash
uv run python -m cli.augmented_generation_cli rag "movies about space exploration"
uv run python -m cli.augmented_generation_cli summarize "movies about space exploration"
uv run python -m cli.augmented_generation_cli citations "movies about space exploration" --limit 3
uv run python -m cli.augmented_generation_cli question "Which movies feature astronauts?" --limit 3
```

## Image search

Prefix: `uv run python -m cli.multimodal_search_cli`

| Command | Purpose |
| --- | --- |
| `verify_image_embedding <image>` | Load the CLIP model and movie text embeddings, then print the local image embedding's dimension count. |
| `image_search <image>` | Compare a local image against movie text embeddings and print the top five movies with similarity scores. |

```bash
uv run python -m cli.multimodal_search_cli verify_image_embedding ./path/to/image.jpg
uv run python -m cli.multimodal_search_cli image_search ./path/to/image.jpg
```

Replace the image path with an existing local file. Both commands load the movie dataset and cache its CLIP text embeddings in `cache/docs_clip.pkl`. These commands do not require an OpenRouter key.

## Rewrite a query using an image

`describe_image_cli` uses a local image and a text query to produce a rewritten movie search query through the LLM. It prints the rewritten query; it does not run a search.

Both flags are required:

```bash
uv run python -m cli.describe_image_cli --image ./path/to/image.jpg --query "What movie is this?"
```

The image MIME type is inferred from the filename, with `image/jpeg` as the fallback. This command requires an OpenRouter key and a model route that supports image input.

## Evaluate retrieval

Run RRF searches against `data/golden_dataset.json` and print precision, recall, F1, retrieved titles, and relevant titles for each test case:

```bash
uv run python -m cli.evaluation_cli --limit 5
```

`--limit` defaults to `5` and sets the retrieval cutoff for precision@k and recall@k. RRF uses a fixed rank constant of `60`. This command compares movie titles against the golden dataset without LLM calls.

Current implementation note: F1 calculation raises `ZeroDivisionError` when precision and recall are both zero.

## LLM smoke test

```bash
uv run python -m cli.test_llm
```

Sends a fixed prompt to OpenRouter and prints the response plus prompt and response token counts. It requires `OPENROUTER_API_KEY` and does not accept command-line arguments.
