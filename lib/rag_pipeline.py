from .hybrid_search import rrf_search
from .hybrid_search import format_results_for_llm_evaluation
from .llm_client import run_augmented_query

def run_rag_pipeline(query: str):
    rrf_res = rrf_search(query=query)
    formated_docs = format_results_for_llm_evaluation(rrf_res)
    llm_res = run_augmented_query(chr(10).join(formated_docs), query)

    print("Search Results:")
    for item in rrf_res:
        print(f"  - {item['title']}")

    print("RAG Response:")
    print(f"{llm_res}")
