"""Test DuckDuckGo web search directly."""

from ddgs import DDGS


def test_web_search():
    """Test if DuckDuckGo search works."""
    print("Testing DuckDuckGo search...\n")

    try:
        ddgs = DDGS()
        query = "Apple revenue Q4 2025"

        print(f"Searching for: {query}")
        results = ddgs.text(query, max_results=5)

        if not results:
            print("\n[ERROR] No results returned!")
            return

        print(f"\n[SUCCESS] Found {len(results)} results:\n")

        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            body = result.get("body", "No description")
            url = result.get("href", "No URL")

            print(f"Result {i}:")
            print(f"  Title: {title}")
            print(f"  URL: {url}")
            print(f"  Description: {body[:100]}...")
            print()

    except Exception as e:
        print(f"\n[ERROR] Exception occurred:")
        print(f"  {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    test_web_search()
