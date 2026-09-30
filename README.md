# 🤖 Autonomous Market Research & Content Agent Studio

> **A modular, evidence-oriented AI research pipeline for transforming natural-language research questions into structured market intelligence.**

The **Autonomous Market Research & Content Agent Studio** is a Python-based research automation project designed to investigate how AI-assisted systems can collect live information, filter research signals, analyze unstructured text, discover recurring themes, extract entities and quantitative evidence, and transform those signals into structured research briefs.

The project is being developed incrementally from a foundational NLP and web-research pipeline toward a more autonomous **multi-agent market research architecture**.

---

# 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Why This Project?](#-why-this-project)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Current System](#-current-system)
- [System Architecture](#-system-architecture)
- [Research Workflow](#-research-workflow)
- [Core Components](#-core-components)
- [NLP & Machine Learning Pipeline](#-nlp--machine-learning-pipeline)
- [Research Relevance Analysis](#-research-relevance-analysis)
- [Sentiment Analysis](#-sentiment-analysis)
- [Keyword Extraction](#-keyword-extraction)
- [Entity Extraction](#-entity-extraction)
- [Quantitative Evidence Extraction](#-quantitative-evidence-extraction)
- [Theme Discovery](#-theme-discovery)
- [Research Brief Generation](#-research-brief-generation)
- [Interactive Dashboard](#-interactive-dashboard)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Running the Application](#-running-the-application)
- [Example Research Query](#-example-research-query)
- [Example Output](#-example-output)
- [Evidence & Source Handling](#-evidence--source-handling)
- [Design Principles](#-design-principles)
- [Current Capabilities](#-current-capabilities)
- [Current Limitations](#-current-limitations)
- [Development Roadmap](#-development-roadmap)
- [Future Multi-Agent Architecture](#-future-multi-agent-architecture)
- [Research Integrity](#-research-integrity)
- [Development Status](#-development-status)
- [Author](#-author)
- [License](#-license)

---

# 🔎 Project Overview

Market research typically requires a researcher to perform several separate activities:

1. Define a research question.
2. Search for relevant information.
3. Collect information from multiple sources.
4. Read and filter large amounts of text.
5. Identify recurring concepts and themes.
6. Extract companies, technologies, markets, and other entities.
7. Identify quantitative evidence.
8. Compare signals across sources.
9. Synthesize the findings.
10. Produce a structured research report.

The objective of this project is to investigate how these stages can be progressively automated through a modular AI-assisted pipeline.

The current implementation focuses on the **research intelligence foundation**:

```text
Research Question
        │
        ▼
Live Source Discovery
        │
        ▼
Article Retrieval
        │
        ▼
Content Extraction
        │
        ▼
NLP Processing
        │
        ├───────────────┐
        │               │
        ▼               ▼
 Sentiment        Relevance Analysis
 Analysis              │
        │               │
        └───────┬───────┘
                ▼
      Research Signal Extraction
                │
        ┌───────┼────────┬───────────┐
        ▼       ▼        ▼           ▼
    Keywords  Entities  Themes   Quantitative
                                  Evidence
        │       │        │           │
        └───────┴────────┴───────────┘
                        │
                        ▼
               Research Synthesis
                        │
                        ▼
                Research Brief
                        │
                        ▼
              Streamlit Dashboard
```

# 🧩 Core Components

The project is divided into modular components, with each component responsible for a specific stage of the research pipeline.

## 1. Research Input Layer

Accepts the user's natural-language research question and controls the research configuration.

**Current responsibility:**

- Accept research topic/question
- Configure number of articles
- Trigger the research workflow

---

## 2. Web Research & Scraping Layer

Implemented primarily through `tools/scraper.py`.

**Responsibilities:**

- Search Google News RSS
- Collect article metadata
- Resolve article URLs
- Retrieve publisher pages
- Extract article content
- Fall back to RSS content when required
- Report scraping diagnostics

---

## 3. NLP & Analytical Layer

Implemented primarily through `tools/sentiment.py`.

**Responsibilities:**

- Text preprocessing
- Sentiment analysis
- Research relevance analysis
- Keyword extraction
- Entity extraction
- Numeric claim extraction
- Theme discovery
- Domain/category detection

---

## 4. Research Synthesis Layer

Implemented through `agents/writer.py`.

**Responsibilities:**

- Consume analytical outputs
- Organize research signals
- Structure evidence
- Generate the initial research brief
- Surface research limitations

---

## 5. Presentation & Visualization Layer

Implemented through `app.py`.

**Responsibilities:**

- Streamlit interface
- Research configuration
- Workflow execution
- Sentiment visualization
- Domain/category visualization
- Research brief presentation
- Raw analytical data inspection

---

## 6. Future Agent Orchestration Layer

This is part of the planned architecture rather than the complete current implementation.

The intended architecture will eventually introduce specialized agents for:

- Research planning
- Source discovery
- Evidence validation
- Market analysis
- Research synthesis
- Guardrails and quality control

This layer is intended to transform the current sequential analytical pipeline into a more autonomous multi-agent research workflow.


