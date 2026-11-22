import asyncio
import os
import re
from typing import List

import apple_fm_sdk as fm
import wikipediaapi
from rank_bm25 import BM25Okapi

wiki = wikipediaapi.Wikipedia(
    user_agent=f'{os.environ.get("PROJECT_NAME")} ({os.environ.get("EMAIL")})',
    language="en",
)

MAX_PASSAGE_CHARS = 6000
MAX_ARTICLES = 6


@fm.generable("Wikipedia search queries")
class Queries:
    queries: List[str] = fm.guide(
        "Keyword queries for Wikipedia search, 1-5 words each, from specific to general",
        count=3,
    )


@fm.generable("Answer to a question based on passages")
class Answer:
    found: bool = fm.guide("True only if the passages contain the answer")
    answer: str = fm.guide("The answer in 1-3 sentences, or empty if not found")
    source: str = fm.guide("Title of the article the answer came from, or empty")


QUERY_INSTRUCTIONS = """
You turn questions into Wikipedia search queries. Use keywords, never full sentences.
Example question: "In what year was the Eiffel Tower completed?"
Example queries: "Eiffel Tower", "Eiffel Tower construction", "Gustave Eiffel"
"""

ANSWER_INSTRUCTIONS = """
You answer questions using only the Wikipedia passages given to you.
Each passage starts with its article title in square brackets.
Do not use your own knowledge. If the passages do not contain the answer, set found to false.
If the question is ambiguous, use the most common meaning and say which one you used.
"""


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def search(query: str) -> List[str]:
    return list(wiki.search(query, limit=3).pages.keys())


def fetch_paragraphs(title: str) -> List[tuple[str, str]]:
    page = wiki.page(title)
    if not page.exists():
        return []
    return [(title, p.strip()) for p in page.text.split("\n") if len(p.strip()) > 80]


def top_passages(query_text: str, passages, budget: int):
    bm25 = BM25Okapi([tokenize(p) for _, p in passages])
    scores = bm25.get_scores(tokenize(query_text))
    ranked = sorted(zip(scores, passages), key=lambda x: -x[0])
    chosen, used = [], 0
    for score, (title, p) in ranked:
        if score <= 0 or used + len(p) > budget:
            continue
        chosen.append((title, p))
        used += len(p)
    return chosen


async def make_queries(question: str) -> List[str]:
    session = fm.LanguageModelSession(instructions=QUERY_INSTRUCTIONS)
    result = await session.respond(question, generating=Queries)
    return [q.strip().strip("\"'") for q in result.queries]


async def answer(question: str) -> str:
    queries = await make_queries(question) + [question]
    print(f"[queries] {queries}")

    results = await asyncio.gather(*(asyncio.to_thread(search, q) for q in queries))
    titles = list(dict.fromkeys(t for r in results for t in r))[:MAX_ARTICLES]
    print(f"[titles] {titles}")
    if not titles:
        return "I could not find this on Wikipedia."

    paragraph_lists = await asyncio.gather(*(asyncio.to_thread(fetch_paragraphs, t) for t in titles))
    passages = [p for lst in paragraph_lists for p in lst]
    if not passages:
        return "I could not find this on Wikipedia."

    best = top_passages(question + " " + " ".join(queries), passages, MAX_PASSAGE_CHARS)
    context = "\n\n".join(f"[{t}] {p}" for t, p in best)

    session = fm.LanguageModelSession(instructions=ANSWER_INSTRUCTIONS)
    result = await session.respond(
        f"Passages:\n{context}\n\nQuestion: {question}",
        generating=Answer,
    )
    if not result.found:
        return "I could not find this on Wikipedia."
    return f"{result.answer}\nSource: {result.source}"


async def main():
    model = fm.SystemLanguageModel()
    ok, reason = model.is_available()
    if not ok:
        print(f"Model not available: {reason}")
        return
    while True:
        task = input(">")
        print(await answer(task))


asyncio.run(main())