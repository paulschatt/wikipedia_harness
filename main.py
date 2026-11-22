import asyncio
import os

import apple_fm_sdk as fm
import wikipediaapi

wiki = wikipediaapi.AsyncWikipedia(
    user_agent=f'{os.environ.get("PROJECT_NAME")} ({os.environ.get("EMAIL")})',
    language="en",
)

MAX_ARTICLE_CHARS = 3000

INSTRUCTIONS = """
You answer questions using Wikipedia. Use the tools to find facts.

How to work:
1. Decide which facts you need.
2. Call search_wikipedia with 1-5 general keywords. Never search for the whole question.
3. If a search returns NO RESULTS, that is normal. Search again with fewer or broader keywords.
   Try at least 3 different searches before giving up.
4. Call get_article on the most relevant title. If it lacks the answer, try another title or search.
5. Answer only from text you retrieved, not from memory.

Answer in 1-3 sentences, then a line "Sources: <article titles>".
Only if 3 different searches and the articles you read do not contain the answer,
reply: "I could not find this on Wikipedia."
If the question is ambiguous, pick the most common meaning and say which one you used.
"""


class SearchWikipedia(fm.Tool):
    name = "search_wikipedia"
    description = (
        "Finds Wikipedia articles that might contain the answer. Use this first. "
        "Returns up to 5 article titles, one per line, or NO RESULTS."
    )

    @fm.generable("Search parameters")
    class Arguments:
        query: str = fm.guide(
            "1-5 general keywords, no quotes, e.g. Eiffel Tower height. Not a full question."
        )

    @property
    def arguments_schema(self) -> fm.GenerationSchema:
        return self.Arguments.generation_schema()

    async def call(self, args: fm.GeneratedContent) -> str:
        query = args.value(str, for_property="query").strip().strip("\"'")
        results = await wiki.search(query, limit=5)
        titles = list(results.pages.keys())
        print(f"[search] {query!r} -> {titles}")
        if not titles:
            return f"NO RESULTS for '{query}'. Search again with fewer or broader keywords."
        return "\n".join(titles)


class GetArticle(fm.Tool):
    name = "get_article"
    description = (
        "Returns the beginning of one Wikipedia article. "
        "Use after search_wikipedia. Returns NOT FOUND if the title does not exist."
    )

    @fm.generable("Article parameters")
    class Arguments:
        title: str = fm.guide("An article title copied exactly from search_wikipedia results")

    @property
    def arguments_schema(self) -> fm.GenerationSchema:
        return self.Arguments.generation_schema()

    async def call(self, args: fm.GeneratedContent) -> str:
        title = args.value(str, for_property="title").strip().strip("\"'")
        page = await wiki.page(title)
        print(f"[get_article] {title!r} exists={page.exists}")  # debug trace
        if not page.exists:
            return f"NOT FOUND: '{title}'. Use a title exactly as search_wikipedia returned it."
        text = page.text
        if len(text) > MAX_ARTICLE_CHARS:
            text = text[:MAX_ARTICLE_CHARS] + "\n[...article truncated]"
        return text


async def answer(question: str) -> str:
    session = fm.LanguageModelSession(
        instructions=INSTRUCTIONS,
        tools=[SearchWikipedia(), GetArticle()],
    )
    return await session.respond(question)


async def main():
    model = fm.SystemLanguageModel()
    ok, reason = model.is_available()
    if not ok:
        print(f"Model not available: {reason}")
        return

    while True:
        question = input("> ")
        print(await answer(question))


asyncio.run(main())