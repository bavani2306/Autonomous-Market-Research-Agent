# tools/sentiment.py

import re
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ============================================================
# CONFIGURATION
# ============================================================

GENERIC_RESEARCH_TERMS = {
    "market",
    "markets",
    "marketplace",
    "size",
    "share",
    "growth",
    "growing",
    "trend",
    "trends",
    "analysis",
    "report",
    "reports",
    "forecast",
    "forecasting",
    "outlook",
    "industry",
    "industries",
    "global",
    "future",
    "technology",
    "technologies",
    "development",
    "developments",
    "sector",
    "sectors",
    "business",
    "businesses",
    "demand",
    "sales",
    "revenue",
    "value",
    "period",
    "year",
    "years",
    "million",
    "billion",
    "cagr",
}


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
    "will",
    "how",
    "what",
    "which",
    "why",
    "when",
    "where",
    "who",
    "than",
    "over",
    "under",
    "between",
    "through",
    "during",
    "about",
    "after",
    "before",
    "up",
    "down",
}


SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Normalize text for NLP processing.
    """

    if text is None:
        return ""

    text = str(text)

    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(
        r"http\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^A-Za-z0-9\s\-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# QUERY PROCESSING
# ============================================================

def extract_query_terms(topic):
    """
    Extract meaningful terms directly from the user's research
    question.

    No domain-specific terms are hardcoded.
    """

    words = re.findall(
        r"[A-Za-z0-9]+",
        str(topic).lower()
    )

    terms = []

    for word in words:

        if word in STOPWORDS:
            continue

        if len(word) < 3:
            continue

        if word not in terms:
            terms.append(word)

    return terms


def extract_query_phrases(topic):
    """
    Extract only contiguous phrases that actually occur in the
    original research question.

    This prevents artificial phrases such as:

        adoption battery
        trends india

    from being created.
    """

    words = re.findall(
        r"[A-Za-z0-9]+",
        str(topic).lower()
    )

    phrases = []

    for n in (2, 3):

        for i in range(
            len(words) - n + 1
        ):

            chunk = words[
                i:i + n
            ]

            # Do not create phrases containing stopwords.
            if any(
                word in STOPWORDS
                for word in chunk
            ):
                continue

            if any(
                len(word) < 3
                for word in chunk
            ):
                continue

            phrase = " ".join(chunk)

            if phrase not in phrases:
                phrases.append(phrase)

    return phrases


def get_specific_query_terms(query_terms):
    """
    Remove generic research vocabulary.

    This is domain-agnostic.

    For example, terms such as:

        market
        growth
        trends
        analysis

    are weak topic identifiers.

    The actual subject terms remain active.
    """

    return [
        term
        for term in query_terms
        if term not in GENERIC_RESEARCH_TERMS
    ]


# ============================================================
# SENTIMENT
# ============================================================

def get_sentiment(text):
    """
    Calculate VADER sentiment.
    """

    text = clean_text(text)

    scores = SENTIMENT_ANALYZER.polarity_scores(
        text
    )

    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"

    elif compound <= -0.05:
        label = "Negative"

    else:
        label = "Neutral"

    return {
        "label": label,
        "score": round(
            compound,
            4
        ),
        "positive": round(
            scores["pos"],
            4
        ),
        "negative": round(
            scores["neg"],
            4
        ),
        "neutral": round(
            scores["neu"],
            4
        ),
    }


# ============================================================
# MATCHING HELPERS
# ============================================================

def normalize_for_matching(text):
    return clean_text(text).lower()


def term_coverage(text, terms):
    """
    Calculate how many supplied terms occur in the text.
    """

    if not terms:
        return 0.0, []

    text = normalize_for_matching(
        text
    )

    matched = []

    for term in terms:

        pattern = (
            r"\b"
            + re.escape(term)
            + r"\b"
        )

        if re.search(
            pattern,
            text
        ):
            matched.append(term)

    coverage = (
        len(matched) / len(terms)
    )

    return coverage, matched


def phrase_coverage(text, phrases):
    """
    Calculate how many real query phrases occur in the text.
    """

    if not phrases:
        return 0.0, []

    text = normalize_for_matching(
        text
    )

    matched = []

    for phrase in phrases:

        if phrase in text:
            matched.append(
                phrase
            )

    coverage = (
        len(matched) / len(phrases)
    )

    return coverage, matched


# ============================================================
# RELEVANCE MODEL
# ============================================================

def calculate_relevance_scores(
    documents,
    topic
):
    """
    Topic-agnostic research relevance model.

    Signals:

        1. TF-IDF similarity
        2. Specific query-term coverage
        3. Specific term coverage across full article
        4. Meaningful query phrase coverage
        5. Title-specificity
        6. Title phrase matching

    The model intentionally does NOT contain EV-specific
    vocabulary.
    """

    if not documents:
        return []

    query_terms = extract_query_terms(
        topic
    )

    query_phrases = extract_query_phrases(
        topic
    )

    specific_terms = get_specific_query_terms(
        query_terms
    )

    prepared_articles = []

    for article in documents:

        if isinstance(
            article,
            dict
        ):

            title = clean_text(
                article.get(
                    "title",
                    ""
                )
            )

            content = clean_text(
                article.get(
                    "content",
                    ""
                )
            )

        else:

            title = ""

            content = clean_text(
                article
            )

        # Focus the relevance model on the opening section
        # rather than the entire webpage.
        lead = content[:4000]

        prepared_articles.append(
            {
                "title": title,
                "content": content,
                "lead": lead,
            }
        )

    # ========================================================
    # TF-IDF
    # ========================================================

    corpus = []

    for article in prepared_articles:

        corpus.append(
            (
                article["title"]
                + " "
                + article["lead"]
            )
        )

    query_text = clean_text(
        topic
    )

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )

        matrix = vectorizer.fit_transform(
            corpus + [query_text]
        )

        query_vector = matrix[-1]

        document_vectors = matrix[:-1]

        similarities = cosine_similarity(
            document_vectors,
            query_vector
        ).flatten()

    except Exception:

        similarities = np.zeros(
            len(prepared_articles)
        )

    # ========================================================
    # DEBUG
    # ========================================================

    print("\n" + "=" * 70)
    print("RELEVANCE MODEL DEBUG")
    print("=" * 70)

    print("\nResearch topic:")
    print(topic)

    print("\nQuery terms:")
    print(query_terms)

    print("\nSpecific query terms:")
    print(specific_terms)

    print("\nQuery phrases:")
    print(query_phrases)

    print()

    results = []

    # ========================================================
    # ARTICLE SCORING
    # ========================================================

    for index, article in enumerate(
        prepared_articles
    ):

        title = article["title"]

        content = article["content"]

        lead = article["lead"]

        # ----------------------------------------------------
        # Specific term coverage
        # ----------------------------------------------------

        lead_specific_score, lead_specific_matches = (
            term_coverage(
                lead,
                specific_terms
            )
        )

        full_specific_score, full_specific_matches = (
            term_coverage(
                content,
                specific_terms
            )
        )

        # ----------------------------------------------------
        # Query-term coverage
        # ----------------------------------------------------

        query_score, query_matches = (
            term_coverage(
                lead,
                query_terms
            )
        )

        # ----------------------------------------------------
        # Query phrase coverage
        # ----------------------------------------------------

        lead_phrase_score, lead_phrase_matches = (
            phrase_coverage(
                lead,
                query_phrases
            )
        )

        title_phrase_score, title_phrase_matches = (
            phrase_coverage(
                title,
                query_phrases
            )
        )

        # ----------------------------------------------------
        # Title-specificity
        # ----------------------------------------------------

        title_specific_score, title_specific_matches = (
            term_coverage(
                title,
                specific_terms
            )
        )

        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        tfidf_score = float(
            similarities[index]
        )

        # ----------------------------------------------------
        # Weighted relevance score
        # ----------------------------------------------------
        #
        # Title specificity is intentionally strong because
        # the article headline usually describes the subject.
        #
        # TF-IDF remains the largest single signal.
        # ----------------------------------------------------

        score = (
            (tfidf_score * 0.35)
            + (lead_specific_score * 0.20)
            + (full_specific_score * 0.10)
            + (lead_phrase_score * 0.10)
            + (title_specific_score * 0.20)
            + (title_phrase_score * 0.05)
        )

        # ----------------------------------------------------
        # Prevent generic lexical overlap from creating a
        # strong relevance score when no specific topic term
        # appears.
        # ----------------------------------------------------

        if specific_terms:

            if (
                not lead_specific_matches
                and not title_specific_matches
            ):

                score *= 0.25

        score = min(
            max(
                score,
                0.0
            ),
            1.0
        )

        score = round(
            score,
            4
        )

        # ----------------------------------------------------
        # Debug
        # ----------------------------------------------------

        print(
            f"[{index + 1}] "
            f"{score:.3f} | "
            f"{title}"
        )

        print(
            f"    TF-IDF: "
            f"{tfidf_score:.3f}"
        )

        print(
            f"    Specific lead matches: "
            f"{lead_specific_matches}"
        )

        print(
            f"    Specific title matches: "
            f"{title_specific_matches}"
        )

        print(
            f"    Query matches: "
            f"{query_matches}"
        )

        print(
            f"    Lead phrases: "
            f"{lead_phrase_matches}"
        )

        print(
            f"    Title phrases: "
            f"{title_phrase_matches}"
        )

        print()

        results.append(
            {
                "score": score,
                "matched_terms": query_matches,
                "specific_matches": lead_specific_matches,
                "full_specific_matches": full_specific_matches,
                "title_matches": title_specific_matches,
                "matched_phrases": lead_phrase_matches,
                "title_phrases": title_phrase_matches,
            }
        )

    print("=" * 70)
    print(
        "END RELEVANCE DEBUG"
    )
    print("=" * 70)
    print()

    return results


def classify_relevance(score):
    """
    Convert numerical relevance into a readable category.
    """

    if score >= 0.55:
        return "High"

    if score >= 0.30:
        return "Moderate"

    if score >= 0.15:
        return "Low"

    return "Peripheral"


# ============================================================
# DOMAIN EXTRACTION
# ============================================================

def extract_domains(
    title,
    content
):
    """
    Preserve the domain/category field expected by the existing
    Streamlit dashboard.

    This is intentionally generic rather than EV-specific.

    Domains are inferred from recurring meaningful terms in the
    article rather than from a hardcoded EV category list.
    """

    combined = clean_text(
        title + " " + content
    ).lower()

    words = re.findall(
        r"\b[a-zA-Z]{4,}\b",
        combined
    )

    filtered = []

    for word in words:

        if word in STOPWORDS:
            continue

        if word in GENERIC_RESEARCH_TERMS:
            continue

        filtered.append(word)

    counts = Counter(
        filtered
    )

    domains = [
        word
        for word, count
        in counts.most_common(5)
        if count >= 2
    ]

    return ", ".join(
        domains[:5]
    )


# ============================================================
# ENTITY EXTRACTION
# ============================================================

def extract_entities(text):
    """
    Lightweight named-entity style extraction based on
    capitalization patterns.
    """

    text = clean_text(
        text
    )

    candidates = re.findall(
        r"\b[A-Z][A-Za-z0-9&\-]{2,}"
        r"(?:\s+[A-Z][A-Za-z0-9&\-]{2,}){0,3}",
        text
    )

    entities = []

    for candidate in candidates:

        candidate = candidate.strip()

        lower = candidate.lower()

        if lower in STOPWORDS:
            continue

        if lower in GENERIC_RESEARCH_TERMS:
            continue

        if len(candidate) < 3:
            continue

        if candidate in {
            "The",
            "This",
            "These",
            "That",
            "With",
            "Between",
            "From",
            "After",
            "Before",
            "India",
        }:
            continue

        if candidate not in entities:

            entities.append(
                candidate
            )

    return entities[:20]


# ============================================================
# NUMERIC CLAIM EXTRACTION
# ============================================================

def extract_numeric_claims(text):
    """
    Extract potentially useful quantitative evidence.
    """

    text = clean_text(
        text
    )

    patterns = [
        r"\$[\d,.]+\s*"
        r"(?:billion|million|trillion)?",

        r"₹[\d,.]+\s*"
        r"(?:crore|lakh|million|billion)?",

        r"\b\d+(?:\.\d+)?%\b",

        r"\b\d+(?:\.\d+)?\s*"
        r"(?:million|billion|trillion)\b",

        r"\b\d+(?:\.\d+)?\s*"
        r"(?:crore|lakh)\b",

        r"\b\d+(?:\.\d+)?\s*"
        r"(?:GWh|MWh|kWh)\b",

        r"\b\d+(?:\.\d+)?\s*"
        r"(?:GW|MW|kW)\b",
    ]

    claims = []

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for match in matches:

            cleaned = match.strip()

            if cleaned not in claims:

                claims.append(
                    cleaned
                )

    return claims[:25]


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

def extract_keywords(
    text,
    top_n=15
):
    """
    Extract meaningful TF-IDF keywords.
    """

    text = clean_text(
        text
    )

    if not text:
        return []

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=100,
            sublinear_tf=True,
        )

        matrix = vectorizer.fit_transform(
            [text]
        )

        feature_names = (
            vectorizer
            .get_feature_names_out()
        )

        scores = matrix.toarray()[0]

        ranked = sorted(
            zip(
                feature_names,
                scores
            ),
            key=lambda item: item[1],
            reverse=True
        )

        keywords = []

        for term, score in ranked:

            if score <= 0:
                continue

            parts = term.split()

            # Skip phrases made entirely of generic research
            # vocabulary.
            if all(
                part.lower()
                in GENERIC_RESEARCH_TERMS
                for part in parts
            ):
                continue

            if term not in keywords:

                keywords.append(
                    term
                )

            if len(keywords) >= top_n:
                break

        return keywords

    except Exception:

        return []


# ============================================================
# THEME DISCOVERY
# ============================================================

def discover_themes(
    documents,
    n_clusters=3
):
    """
    Unsupervised theme discovery using TF-IDF + KMeans.

    Themes are signals, not final conclusions.
    """

    if not documents:
        return []

    texts = []

    for article in documents:

        if isinstance(
            article,
            dict
        ):

            text = (
                clean_text(
                    article.get(
                        "title",
                        ""
                    )
                )
                + " "
                + clean_text(
                    article.get(
                        "content",
                        ""
                    )
                )
            )

        else:

            text = clean_text(
                article
            )

        texts.append(
            text
        )

    if len(texts) < 2:
        return []

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=150,
            min_df=1,
            sublinear_tf=True,
        )

        matrix = vectorizer.fit_transform(
            texts
        )

        cluster_count = min(
            n_clusters,
            len(texts)
        )

        if cluster_count < 2:
            return []

        model = KMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(
            matrix
        )

        feature_names = (
            vectorizer
            .get_feature_names_out()
        )

        themes = []

        for cluster_id in range(
            cluster_count
        ):

            center = (
                model
                .cluster_centers_[
                    cluster_id
                ]
            )

            top_indices = (
                center.argsort()[::-1][:15]
            )

            cluster_terms = []

            for index in top_indices:

                term = feature_names[
                    index
                ]

                parts = term.split()

                if all(
                    part.lower()
                    in GENERIC_RESEARCH_TERMS
                    for part in parts
                ):
                    continue

                if term not in cluster_terms:

                    cluster_terms.append(
                        term
                    )

                if len(cluster_terms) >= 5:
                    break

            article_count = int(
                np.sum(
                    labels
                    == cluster_id
                )
            )

            if cluster_terms:

                themes.append(
                    {
                        "theme": " / ".join(
                            cluster_terms[:3]
                        ),
                        "keywords": cluster_terms,
                        "article_count": article_count,
                    }
                )

        themes.sort(
            key=lambda item:
            item["article_count"],
            reverse=True
        )

        return themes

    except Exception:

        return []


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_articles(
    articles,
    topic
):
    """
    Main NLP analysis function.

    IMPORTANT:
    This function intentionally returns a Pandas DataFrame because
    the existing app.py expects:

        df['sentiment']
        df['domains']
        df['score']
        df['title']
        df['link']

    Additional analytical information is preserved in extra
    DataFrame columns.
    """

    if not articles:

        return pd.DataFrame(
            columns=[
                "title",
                "link",
                "published",
                "content",
                "source",
                "domains",
                "sentiment",
                "score",
                "sentiment_positive",
                "sentiment_negative",
                "sentiment_neutral",
                "relevance",
                "relevance_score",
                "matched_terms",
                "specific_matches",
                "title_matches",
                "matched_phrases",
                "title_phrases",
                "entities",
                "numeric_claims",
                "keywords",
            ]
        )

    # ========================================================
    # RELEVANCE
    # ========================================================

    relevance_results = (
        calculate_relevance_scores(
            articles,
            topic
        )
    )

    # ========================================================
    # ARTICLE ANALYSIS
    # ========================================================

    rows = []

    all_texts = []

    entity_counter = Counter()

    keyword_counter = Counter()

    relevance_counter = Counter()

    for index, article in enumerate(
        articles
    ):

        title = article.get(
            "title",
            ""
        )

        content = article.get(
            "content",
            ""
        )

        link = article.get(
            "link",
            ""
        )

        published = article.get(
            "published",
            ""
        )

        source = article.get(
            "source",
            ""
        )

        combined_text = (
            clean_text(title)
            + " "
            + clean_text(content)
        )

        all_texts.append(
            combined_text
        )

        # ----------------------------------------------------
        # Sentiment
        # ----------------------------------------------------

        sentiment = get_sentiment(
            combined_text
        )

        # ----------------------------------------------------
        # Domains
        # ----------------------------------------------------

        domains = extract_domains(
            title,
            content
        )

        # ----------------------------------------------------
        # Entities
        # ----------------------------------------------------

        entities = extract_entities(
            combined_text
        )

        for entity in entities:

            entity_counter[
                entity
            ] += 1

        # ----------------------------------------------------
        # Numeric claims
        # ----------------------------------------------------

        numeric_claims = (
            extract_numeric_claims(
                combined_text
            )
        )

        # ----------------------------------------------------
        # Keywords
        # ----------------------------------------------------

        keywords = extract_keywords(
            combined_text
        )

        for keyword in keywords:

            keyword_counter[
                keyword
            ] += 1

        # ----------------------------------------------------
        # Relevance
        # ----------------------------------------------------

        relevance_info = (
            relevance_results[index]
        )

        relevance_score = (
            relevance_info["score"]
        )

        relevance_label = (
            classify_relevance(
                relevance_score
            )
        )

        relevance_counter[
            relevance_label
        ] += 1

        # ----------------------------------------------------
        # Preserve existing expected columns
        # ----------------------------------------------------

        rows.append(
            {
                "title": title,

                "link": link,

                "published": published,

                "content": content,

                "source": source,

                "domains": domains,

                # Existing app.py expects this exact name.
                "sentiment": sentiment[
                    "label"
                ],

                # Existing app.py expects this exact name.
                "score": sentiment[
                    "score"
                ],

                "sentiment_positive": sentiment[
                    "positive"
                ],

                "sentiment_negative": sentiment[
                    "negative"
                ],

                "sentiment_neutral": sentiment[
                    "neutral"
                ],

                "relevance": relevance_label,

                "relevance_score": (
                    relevance_score
                ),

                "matched_terms": (
                    ", ".join(
                        relevance_info[
                            "matched_terms"
                        ]
                    )
                ),

                "specific_matches": (
                    ", ".join(
                        relevance_info[
                            "specific_matches"
                        ]
                    )
                ),

                "title_matches": (
                    ", ".join(
                        relevance_info[
                            "title_matches"
                        ]
                    )
                ),

                "matched_phrases": (
                    ", ".join(
                        relevance_info[
                            "matched_phrases"
                        ]
                    )
                ),

                "title_phrases": (
                    ", ".join(
                        relevance_info[
                            "title_phrases"
                        ]
                    )
                ),

                "entities": (
                    ", ".join(
                        entities
                    )
                ),

                "numeric_claims": (
                    ", ".join(
                        numeric_claims
                    )
                ),

                "keywords": (
                    ", ".join(
                        keywords
                    )
                ),
            }
        )

    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        rows
    )

    # ========================================================
    # GLOBAL THEMES
    # ========================================================

    themes = discover_themes(
        articles,
        n_clusters=min(
            4,
            len(articles)
        )
    )

    # Convert themes into a readable string so the information
    # remains available without changing the app.py contract.

    theme_strings = []

    for theme in themes:

        theme_strings.append(
            theme["theme"]
        )

    # Store global theme information in DataFrame metadata.
    df.attrs["themes"] = themes

    # ========================================================
    # GLOBAL KEYWORDS
    # ========================================================

    top_keywords = [
        keyword
        for keyword, count
        in keyword_counter.most_common(
            20
        )
    ]

    df.attrs[
        "top_keywords"
    ] = top_keywords

    # ========================================================
    # GLOBAL ENTITIES
    # ========================================================

    top_entities = [
        entity
        for entity, count
        in entity_counter.most_common(
            20
        )
    ]

    df.attrs[
        "top_entities"
    ] = top_entities

    # ========================================================
    # GLOBAL NUMERIC EVIDENCE
    # ========================================================

    numeric_evidence = []

    for value in df[
        "numeric_claims"
    ].dropna():

        for claim in str(
            value
        ).split(", "):

            if (
                claim
                and claim
                not in numeric_evidence
            ):

                numeric_evidence.append(
                    claim
                )

    df.attrs[
        "numeric_claims"
    ] = numeric_evidence[:30]

    # ========================================================
    # RELEVANCE SUMMARY
    # ========================================================

    df.attrs[
        "relevance_summary"
    ] = {
        "High": relevance_counter.get(
            "High",
            0
        ),
        "Moderate": relevance_counter.get(
            "Moderate",
            0
        ),
        "Low": relevance_counter.get(
            "Low",
            0
        ),
        "Peripheral": relevance_counter.get(
            "Peripheral",
            0
        ),
    }

    # ========================================================
    # SENTIMENT SUMMARY
    # ========================================================

    sentiment_counts = (
        df["sentiment"]
        .value_counts()
        .to_dict()
    )

    df.attrs[
        "sentiment_summary"
    ] = {
        "Positive": sentiment_counts.get(
            "Positive",
            0
        ),
        "Negative": sentiment_counts.get(
            "Negative",
            0
        ),
        "Neutral": sentiment_counts.get(
            "Neutral",
            0
        ),
    }

    # ========================================================
    # QUERY INFORMATION
    # ========================================================

    df.attrs[
        "research_topic"
    ] = topic

    df.attrs[
        "query_terms"
    ] = extract_query_terms(
        topic
    )

    df.attrs[
        "query_phrases"
    ] = extract_query_phrases(
        topic
    )

    df.attrs[
        "specific_query_terms"
    ] = get_specific_query_terms(
        extract_query_terms(
            topic
        )
    )

    # ========================================================
    # FINAL DATAFRAME
    # ========================================================

    return df