import os
from dotenv import load_dotenv
from openai import OpenAI
from openai import APIStatusError
from openai import APIConnectionError
import time
import json


load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

def spell_check(query: str) -> str:
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
    return __exec_and_llm_response(prompt)


def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
    return __exec_and_llm_response(prompt)


def expand_query(query: str) -> str:
    prompt = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"
- "math" -> "I.Q. mathematics numbers equation formula logic"

User query: "{query}"
"""
    return __exec_and_llm_response(prompt)


def rerank_doc(doc: dict, query: str) -> str:
    prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""
    return __exec_and_llm_response(prompt=prompt)


def rerank_doc_batch(docs: list[dict], query: str) -> list[int]:
    prompt = f"""Rank the movies listed below by relevance to the following search query.
Query: "{query}"
Movies:
{docs}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

    res = __exec_and_llm_response(prompt=prompt)
    try:
        ints = json.loads(res)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"error converting the LLM response '{res}' to list of integers: {e}")
        return []


def evaluate_results(query: str, formatted_results: list[str]) -> list[int]:
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents.
Return a valid JSON list, nothing else. For example:
[2, 0, 3, 2, 0, 1]"""
    resp = __exec_and_llm_response(prompt)
    print(f"--> DEBUG: LLM-RESPONSE='{resp}'")
    ints = json.loads(resp)
    return ints


def run_augmented_query(formatted_docs: str, query: str) -> str:
    prompt = f"""You are a RAG agent for a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{formatted_docs}

Answer only as bullet-pointed list and nothing else, like:
- movie 1
- movie 2
- movie 3"""
    return __exec_and_llm_response(prompt)


def run_summarization_query(query: str, formatted_docs: str) -> str:
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to a movie streaming service users.

Query: {query}

Search results:
{formatted_docs}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
    return __exec_and_llm_response(prompt)


def __exec_and_llm_response(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    #model = "openrouter/free"
    model = "~deepseek/deepseek-v4-flash-latest"

    try:
        response = client.chat.completions.create(messages=messages, model=model)
        return response.choices[0].message.content.strip()
    except APIStatusError as e:
        print(f"API Error (Status Code {e.status_code}): {e.message}")
        return f"Error: API returned status code {e.status_code}"
    except APIConnectionError as e:
        print(f"Connection Error: {e.__cause__}")
        return "Error: Could not connect to the server"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "Error: An unexpected error occurred"
    finally:
        if model == "openrouter/free":
            print(f"sleeping to avoid llm rate-limiting for subsequent calls for model {model}")
            time.sleep(3)

def test():
    messages = [
        {
            "role": "user",
            "content": "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
        }
    ]

    model = "openrouter/free"

    response = client.chat.completions.create(messages=messages, model=model)
    print(response.choices[0].message.content)
    print(f"Prompt tokens: {response.usage.prompt_tokens}")
    print(f"Response tokens: {response.usage.completion_tokens}")