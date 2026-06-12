from langchain_community.tools import DuckDuckGoSearchResults

_search = DuckDuckGoSearchResults(num_results=5, output_format="list")


def web_search(query: str) -> tuple[str, list[str]]:
    """Return (formatted_results, list_of_source_urls)."""
    try:
        results = _search.run(query)

        if not isinstance(results, list):
            return str(results), []

        lines = []
        urls = []
        for r in results:
            title = r.get("title", "")
            snippet = r.get("snippet", r.get("body", ""))
            url = r.get("link", r.get("href", ""))
            if title or snippet:
                lines.append(f"**{title}**\n{snippet}")
            if url:
                urls.append(url)

        return "\n\n".join(lines), urls

    except Exception as e:
        return f"Search failed: {e}", []
