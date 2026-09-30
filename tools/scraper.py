import feedparser
import requests
from bs4 import BeautifulSoup
from googlenewsdecoder import gnewsdecoder

MAX_ARTICLES = 15
REQUEST_TIMEOUT = 15


def resolve_google_news_url(google_url):
    """
    Convert a Google News RSS article URL into the original publisher URL.
    """
    try:
        result = gnewsdecoder(
            google_url,
            interval=1,
            timeout=REQUEST_TIMEOUT
        )

        if result.get("success"):
            return result.get("decoded_url")

    except Exception as e:
        print(f"  Decoder error: {e}")

    return None


def extract_article_text(url):
    """
    Fetch the publisher page and extract readable article text.
    """
    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/154.0.0.0 Safari/537.36"
                )
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove elements that are normally not article content.
        for tag in soup([
            "script",
            "style",
            "noscript",
            "nav",
            "header",
            "footer",
            "aside",
            "form",
            "svg"
        ]):
            tag.decompose()

        # First preference: semantic <article>
        article = soup.find("article")

        if article:
            paragraphs = article.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        text_parts = []

        for paragraph in paragraphs:
            text = paragraph.get_text(" ", strip=True)

            if len(text) >= 40:
                text_parts.append(text)

        text = "\n".join(text_parts)

        # Avoid treating tiny/empty pages as successful extraction.
        if len(text) >= 500:
            return text, "full_article"

        # Second attempt: <main>
        main = soup.find("main")

        if main:
            paragraphs = main.find_all("p")

            text_parts = []

            for paragraph in paragraphs:
                text = paragraph.get_text(" ", strip=True)

                if len(text) >= 40:
                    text_parts.append(text)

            text = "\n".join(text_parts)

            if len(text) >= 500:
                return text, "full_article_main"

        return None, "page_too_short"

    except Exception as e:
        print(f"  Article extraction error: {e}")
        return None, "fetch_failed"


def scrape_news(query, max_results=MAX_ARTICLES):
    """
    Search Google News RSS, resolve publisher URLs,
    and extract actual article content.
    """

    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={requests.utils.quote(query)}"
        "&hl=en-IN"
        "&gl=IN"
        "&ceid=IN:en"
    )

    print("\n" + "=" * 70)
    print("MARKET RESEARCH SCRAPER")
    print("=" * 70)

    print(f"Research query: {query}")
    print(f"Maximum articles: {max_results}")

    # ---------------------------------------------------------
    # STEP 1: Get Google News RSS results
    # ---------------------------------------------------------

    feed = feedparser.parse(rss_url)

    entries = feed.entries[:max_results]

    print(f"\nRSS results found: {len(entries)}")

    if not entries:
        print("No Google News results found.")
        return []

    articles = []

    # ---------------------------------------------------------
    # STEP 2: Resolve each Google News URL
    # ---------------------------------------------------------

    for index, entry in enumerate(entries, start=1):

        title = entry.get("title", "").strip()
        google_url = entry.get("link", "").strip()
        published = entry.get("published", "").strip()

        print("\n" + "-" * 70)
        print(f"Article {index}/{len(entries)}")
        print(f"Title: {title}")

        # Google News RSS description
        rss_summary = entry.get("summary", "").strip()

        print(f"Google URL: {google_url}")

        publisher_url = resolve_google_news_url(google_url)

        if publisher_url:
            print(f"Publisher URL: {publisher_url}")
        else:
            print("Publisher URL: FAILED")

        # -----------------------------------------------------
        # STEP 3: Fetch actual publisher article
        # -----------------------------------------------------

        content = None
        content_source = None
        fetch_status = None

        if publisher_url:

            content, content_source = extract_article_text(
                publisher_url
            )

            if content:
                fetch_status = "success"
                print(
                    f"Content extracted: {len(content)} characters"
                )
            else:
                fetch_status = "content_unavailable"
                print("Publisher page found, but article text unavailable.")

        else:
            fetch_status = "url_resolution_failed"

        # -----------------------------------------------------
        # STEP 4: RSS fallback
        # -----------------------------------------------------

        if not content:

            # Google News RSS summaries are usually short.
            # We keep them only as fallback evidence.
            content = rss_summary

            if content:
                content_source = "rss_summary"
                print("Using RSS summary fallback.")
            else:
                content_source = "none"
                print("No usable content available.")

        # -----------------------------------------------------
        # STEP 5: Determine source name
        # -----------------------------------------------------

        source = ""

        if hasattr(entry, "source"):
            source_data = entry.source

            if isinstance(source_data, dict):
                source = source_data.get("title", "")

        if not source:
            source = "Unknown source"

        # -----------------------------------------------------
        # STEP 6: Store normalized article
        # -----------------------------------------------------

        articles.append({
            "title": title,
            "link": publisher_url or google_url,
            "published": published,
            "content": content,
            "source": source,
            "content_source": content_source,
            "fetch_status": fetch_status,
            "google_news_url": google_url
        })

    # ---------------------------------------------------------
    # STEP 7: Diagnostic summary
    # ---------------------------------------------------------

    full_articles = sum(
        1
        for article in articles
        if article["content_source"] in [
            "full_article",
            "full_article_main"
        ]
    )

    rss_fallbacks = sum(
        1
        for article in articles
        if article["content_source"] == "rss_summary"
    )

    url_failures = sum(
        1
        for article in articles
        if article["fetch_status"] == "url_resolution_failed"
    )

    content_failures = sum(
        1
        for article in articles
        if article["fetch_status"] == "content_unavailable"
    )

    print("\n" + "=" * 70)
    print("SCRAPER DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print(f"RSS results:                 {len(entries)}")
    print(f"Sources collected:           {len(articles)}")
    print(f"Full article pages:           {full_articles}")
    print(f"RSS summary fallbacks:        {rss_fallbacks}")
    print(f"URL resolution failures:      {url_failures}")
    print(f"Article extraction failures:  {content_failures}")

    print("\nCONTENT SOURCES")

    for index, article in enumerate(articles, start=1):

        preview = article["content"].replace("\n", " ")[:180]

        print(
            f"\n[{index}] {article['source']}"
            f"\n    Content source: {article['content_source']}"
            f"\n    Status: {article['fetch_status']}"
            f"\n    Length: {len(article['content'])}"
            f"\n    Preview: {preview}"
        )

    print("\n" + "=" * 70)

    return articles