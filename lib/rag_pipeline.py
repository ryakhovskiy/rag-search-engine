from .hybrid_search import rrf_search
from .hybrid_search import format_results_for_llm_evaluation
from .llm_client import run_augmented_query
from .llm_client import run_summarization_query
from .llm_client import run_citations_query
from .llm_client import run_question_query

def run_rag_pipeline(query: str):
    print(f"running RAG pipeline for query '{query}'")
    rrf_res = rrf_search(query=query)
    formated_docs = format_results_for_llm_evaluation(rrf_res)
    llm_res = run_augmented_query(chr(10).join(formated_docs), query)

    print("Search Results:")
    for item in rrf_res:
        print(f"  - {item['title']}")

    print("RAG Response:")
    print(f"{llm_res}")

def run_summarization(query: str):
    print(f"running summarization for query '{query}'")
    rrf_res = rrf_search(query=query)
    formated_docs = format_results_for_llm_evaluation(rrf_res)
    summary_res = run_summarization_query(chr(10).join(formated_docs), query)

    print("Search Results:")
    for item in rrf_res:
        print(f"  - {item['title']}")
    print("LLM Summary:")
    print(f"{summary_res}")


def run_citations(query: str, limit: int = 5):
    print(f"running citations for query {query}")
    rrf_res = rrf_search(query=query, limit=limit)
    formated_docs = format_results_for_llm_evaluation(rrf_res)
    citations_res = run_citations_query(query, formated_docs)
    print("Search Results:")
    for item in rrf_res:
        print(f"  - {item['title']}")
    print("LLM Answer:")
    print(f"{citations_res}")


def run_question(question: str, limit: int = 5):
    print(f"running RAG for question {question}")
    rrf_res = rrf_search(query=question, limit=limit)
    formated_docs = format_results_for_llm_evaluation(rrf_res)
    answer = run_question_query(question, formated_docs)
    print("Search Results:")
    for item in rrf_res:
        print(f"  - {item['title']}")
    print("Answer:")
    print(f"{answer}")
