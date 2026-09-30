from collections import Counter
import re


# =========================================================
# TEXT HELPERS
# =========================================================

def clean_text(value):
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value)
    ).strip()


def format_list(items, limit=10):
    cleaned = []

    for item in items:

        item = clean_text(item)

        if not item:
            continue

        if item not in cleaned:
            cleaned.append(item)

        if len(cleaned) >= limit:
            break

    return cleaned


# =========================================================
# DATA EXTRACTION
# =========================================================

def get_top_keywords(df, limit=12):

    keywords = df.attrs.get(
        "top_keywords",
        []
    )

    return format_list(
        keywords,
        limit
    )


def get_top_entities(df, limit=12):

    entities = df.attrs.get(
        "top_entities",
        []
    )

    return format_list(
        entities,
        limit
    )


def get_numeric_claims(df, limit=10):

    claims = df.attrs.get(
        "numeric_claims",
        []
    )

    return format_list(
        claims,
        limit
    )


def get_sentiment_summary(df):

    total = len(df)

    if total == 0:
        return {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "positive_pct": 0,
            "negative_pct": 0,
            "neutral_pct": 0
        }

    positive = int(
        df["sentiment"]
        .str.contains(
            "Optimistic",
            case=False,
            na=False
        )
        .sum()
    )

    negative = int(
        df["sentiment"]
        .str.contains(
            "Cautious|Critical",
            case=False,
            na=False
        )
        .sum()
    )

    neutral = total - positive - negative

    return {
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "positive_pct": round(
            positive / total * 100,
            1
        ),
        "negative_pct": round(
            negative / total * 100,
            1
        ),
        "neutral_pct": round(
            neutral / total * 100,
            1
        )
    }


# =========================================================
# THEME ANALYSIS
# =========================================================

def build_theme_analysis(df):

    if "domains" not in df.columns:
        return []

    theme_counts = Counter(
        clean_text(value)
        for value in df["domains"]
        if clean_text(value)
    )

    results = []

    for theme, count in theme_counts.most_common():

        rows = df[
            df["domains"].astype(str) == str(theme)
        ]

        positive = int(
            rows["sentiment"]
            .str.contains(
                "Optimistic",
                case=False,
                na=False
            )
            .sum()
        )

        negative = int(
            rows["sentiment"]
            .str.contains(
                "Cautious|Critical",
                case=False,
                na=False
            )
            .sum()
        )

        neutral = (
            len(rows)
            - positive
            - negative
        )

        results.append({
            "theme": theme,
            "mentions": count,
            "positive": positive,
            "negative": negative,
            "neutral": neutral
        })

    return results


# =========================================================
# SOURCE EVIDENCE
# =========================================================

def build_source_evidence(df):

    evidence = []

    for _, row in df.iterrows():

        title = clean_text(
            row.get(
                "title",
                ""
            )
        )

        if not title:
            continue

        entities = row.get(
            "entities",
            []
        )

        numeric_claims = row.get(
            "numeric_claims",
            []
        )

        if not isinstance(
            entities,
            list
        ):
            entities = []

        if not isinstance(
            numeric_claims,
            list
        ):
            numeric_claims = []

        evidence.append({

            "title": title,

            "link": clean_text(
                row.get(
                    "link",
                    ""
                )
            ),

            "published": clean_text(
                row.get(
                    "published",
                    ""
                )
            ),

            "sentiment": clean_text(
                row.get(
                    "sentiment",
                    ""
                )
            ),

            "relevance": row.get(
                "relevance",
                0
            ),

            "theme": clean_text(
                row.get(
                    "domains",
                    ""
                )
            ),

            "entities": format_list(
                entities,
                8
            ),

            "numeric_claims": format_list(
                numeric_claims,
                5
            )
        })

    return evidence


# =========================================================
# RELEVANT SOURCES
# =========================================================

def get_relevant_sources(
    evidence,
    minimum_relevance=0.20
):

    relevant = []

    for item in evidence:

        try:
            relevance = float(
                item["relevance"]
            )
        except (
            TypeError,
            ValueError
        ):
            relevance = 0

        if relevance >= minimum_relevance:

            relevant.append(
                item
            )

    return relevant


# =========================================================
# FINDING GENERATION
# =========================================================

def build_findings(
    themes,
    keywords,
    entities,
    numeric_claims,
    evidence
):

    findings = []

    # -----------------------------------------------------
    # THEME FINDING
    # -----------------------------------------------------

    if themes:

        for theme in themes[:4]:

            findings.append({
                "type": "theme",
                "text": (
                    f"The research corpus repeatedly "
                    f"contains the theme "
                    f"**{theme['theme']}**, appearing in "
                    f"**{theme['mentions']}** source(s)."
                )
            })

    # -----------------------------------------------------
    # ENTITY FINDING
    # -----------------------------------------------------

    if entities:

        findings.append({
            "type": "entities",
            "text": (
                "Frequently detected named entities include: "
                + ", ".join(
                    f"**{entity}**"
                    for entity in entities[:8]
                )
                + "."
            )
        })

    # -----------------------------------------------------
    # KEYWORD FINDING
    # -----------------------------------------------------

    if keywords:

        findings.append({
            "type": "keywords",
            "text": (
                "Recurring concepts extracted from the "
                "research corpus include: "
                + ", ".join(
                    f"`{keyword}`"
                    for keyword in keywords[:10]
                )
                + "."
            )
        })

    # -----------------------------------------------------
    # NUMERIC EVIDENCE
    # -----------------------------------------------------

    if numeric_claims:

        findings.append({
            "type": "numeric",
            "text": (
                "The retrieved sources contain "
                "**quantitative claims that require "
                "source-level verification**, including "
                f"{len(numeric_claims)} extracted claim(s)."
            )
        })

    # -----------------------------------------------------
    # RELEVANCE
    # -----------------------------------------------------

    if evidence:

        relevant_count = len(
            get_relevant_sources(
                evidence
            )
        )

        relevance_pct = round(
            relevant_count
            / len(evidence)
            * 100,
            1
        )

        findings.append({
            "type": "relevance",
            "text": (
                f"Using the current lexical relevance "
                f"filter, **{relevant_count} of "
                f"{len(evidence)} sources "
                f"({relevance_pct}%)** were considered "
                f"relevant enough for downstream analysis."
            )
        })

    return findings


# =========================================================
# REPORT GENERATION
# =========================================================

def generate_brief(
    df,
    topic
):

    topic = clean_text(
        topic
    )

    if df.empty:

        return (
            f"# 🎯 MARKET INTELLIGENCE BRIEF\n\n"
            f"## Research Question\n\n"
            f"**{topic}**\n\n"
            "---\n\n"
            "No live research sources were retrieved. "
            "The system will not generate conclusions "
            "without evidence."
        )

    # -----------------------------------------------------
    # COLLECT EVIDENCE
    # -----------------------------------------------------

    keywords = get_top_keywords(
        df
    )

    entities = get_top_entities(
        df
    )

    numeric_claims = get_numeric_claims(
        df
    )

    sentiment = get_sentiment_summary(
        df
    )

    themes = build_theme_analysis(
        df
    )

    evidence = build_source_evidence(
        df
    )

    relevant_sources = get_relevant_sources(
        evidence
    )

    findings = build_findings(
        themes,
        keywords,
        entities,
        numeric_claims,
        evidence
    )

    # -----------------------------------------------------
    # REPORT
    # -----------------------------------------------------

    report = []

    report.append(
        "# 🎯 MARKET INTELLIGENCE BRIEF"
    )

    report.append(
        f"""
## Research Question

**{topic}**
"""
    )

    # -----------------------------------------------------
    # COVERAGE
    # -----------------------------------------------------

    report.append(
        f"""
---

## 1. Research Coverage

The research pipeline retrieved **{len(df)} live source(s)**.

Of these, **{len(relevant_sources)} source(s)** passed the current relevance threshold for downstream research analysis.

### Sentiment Distribution

| Signal | Articles | Share |
|---|---:|---:|
| Optimistic | {sentiment["positive"]} | {sentiment["positive_pct"]}% |
| Cautious / Critical | {sentiment["negative"]} | {sentiment["negative_pct"]}% |
| Neutral / Reporting | {sentiment["neutral"]} | {sentiment["neutral_pct"]}% |

Sentiment is an article-level linguistic signal and should not be interpreted as a direct measurement of market performance.
"""
    )

    # -----------------------------------------------------
    # RESEARCH SIGNALS
    # -----------------------------------------------------

    report.append(
        """
---

## 2. Research Signals
"""
    )

    if findings:

        for finding in findings:

            report.append(
                f"- {finding['text']}"
            )

    else:

        report.append(
            "No sufficiently strong research signals "
            "were extracted."
        )

    # -----------------------------------------------------
    # THEMES
    # -----------------------------------------------------

    report.append(
        """
---

## 3. Detected Research Themes
"""
    )

    if themes:

        for index, theme in enumerate(
            themes[:5],
            start=1
        ):

            report.append(
                f"""
### {index}. {theme['theme']}

- Source count: **{theme['mentions']}**
- Positive sentiment: **{theme['positive']}**
- Critical sentiment: **{theme['negative']}**
- Neutral/reporting: **{theme['neutral']}**
"""
            )

    else:

        report.append(
            "No recurring themes were detected."
        )

    # -----------------------------------------------------
    # ENTITIES
    # -----------------------------------------------------

    report.append(
        """
---

## 4. Detected Entities
"""
    )

    if entities:

        report.append(
            "\n".join(
                f"- **{entity}**"
                for entity in entities
            )
        )

    else:

        report.append(
            "No reliable named entities were extracted."
        )

    # -----------------------------------------------------
    # KEY CONCEPTS
    # -----------------------------------------------------

    report.append(
        """
---

## 5. Recurring Concepts
"""
    )

    if keywords:

        report.append(
            " • ".join(
                f"`{keyword}`"
                for keyword in keywords
            )
        )

    else:

        report.append(
            "No recurring concepts were extracted."
        )

    # -----------------------------------------------------
    # QUANTITATIVE EVIDENCE
    # -----------------------------------------------------

    report.append(
        """
---

## 6. Quantitative Evidence
"""
    )

    if numeric_claims:

        for claim in numeric_claims:

            report.append(
                f"- {claim}"
            )

    else:

        report.append(
            "No numerical claims were extracted "
            "from the available source content."
        )

    # -----------------------------------------------------
    # SOURCE AUDIT
    # -----------------------------------------------------

    report.append(
        """
---

## 7. Source Evidence Audit
"""
    )

    for item in evidence:

        relevance = item[
            "relevance"
        ]

        relevance_text = (
            f"{float(relevance):.2f}"
            if isinstance(
                relevance,
                (int, float)
            )
            else "N/A"
        )

        report.append(
            f"""
### {item['title']}

- **Relevance:** {relevance_text}
- **Theme:** {item['theme'] or 'Not classified'}
- **Sentiment:** {item['sentiment'] or 'Not classified'}
- **Entities:** {', '.join(item['entities']) if item['entities'] else 'None detected'}
"""
        )

        if item["numeric_claims"]:

            report.append(
                "**Extracted quantitative claims:**"
            )

            for claim in item[
                "numeric_claims"
            ]:

                report.append(
                    f"- {claim}"
                )

        if item["link"]:

            report.append(
                f"- [Source]({item['link']})"
            )

    # -----------------------------------------------------
    # LIMITATIONS
    # -----------------------------------------------------

    report.append(
        """
---

## 8. Research Limitations

This system currently performs automated source discovery, lexical relevance analysis, sentiment analysis, keyword extraction, lightweight entity extraction, numerical-claim extraction, and unsupervised theme discovery.

The extracted claims are **not independently verified**. Numerical statements should be checked against their original sources before being used for business decisions.

The current system also does not yet establish causal relationships, calculate market share independently, or determine that one company or technology is a market leader solely from article frequency.
"""
    )

    report.append(
        """
---

*Autonomous Market Intelligence Agent Studio | Evidence-Driven Research Pipeline*
"""
    )

    return "\n".join(
        report
    )