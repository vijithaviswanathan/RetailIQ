import re
import requests
import numpy as np

from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="RetailIQ RAG Decision Assistant")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTEXT_PATH = PROJECT_ROOT / "reports" / "decision_context.md"

from retailiq.config import (
    CHAT_MODEL,
    EMBEDDING_MODEL,
    EMBEDDING_URL,
    CHAT_URL,
)

TOP_K = 3
REQUEST_TIMEOUT = 180
MIN_RELEVANCE_SCORE = 0.50

STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this",
    "what", "which", "when", "where", "who", "how",
    "does", "did", "are", "was", "were", "has", "have",
    "had", "been", "being", "into", "about", "include",
    "including", "using", "used", "their", "there",
    "they", "them", "than", "then", "also", "most",
    "best", "performed", "performance", "explain",
    "compare", "compared", "show", "give", "provide",
    "model", "models", "result", "results",
}


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


def tokenize(text):
    """Extract meaningful lowercase terms for lexical matching."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return {
        word for word in words
        if len(word) > 2 and word not in STOP_WORDS
    }


def split_into_chunks(document):
    """Split Markdown into focused evidence chunks."""
    chunks = []

    parent_heading = "Project Context"
    current_heading = parent_heading
    current_lines = []
    skip_section = False

    def save_chunk():
        body = "\n".join(current_lines).strip()

        if body and not skip_section:
            chunks.append({
                "heading": current_heading,
                "text": f"{current_heading}\n\n{body}",
            })

    for line in document.splitlines():
        match = re.match(r"^(#{2,3})\s+(.+)$", line)

        if not match:
            if not skip_section:
                current_lines.append(line)
            continue

        level = len(match.group(1))
        title = match.group(2).strip()

        save_chunk()
        current_lines = []

        if level == 2:
            parent_heading = title
            current_heading = title
            skip_section = title.lower().endswith(
                "assistant response rules"
            )
        else:
            current_heading = f"{parent_heading} > {title}"

    save_chunk()

    if not chunks:
        raise ValueError("No evidence chunks could be created.")

    return chunks


def get_embeddings(texts):
    """Generate embeddings using Ollama."""
    response = requests.post(
        EMBEDDING_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": texts,
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    embeddings = response.json().get("embeddings")

    if not embeddings or len(embeddings) != len(texts):
        raise ValueError("Ollama returned invalid embeddings.")

    vectors = np.asarray(embeddings, dtype=np.float32)

    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Invalid embedding vectors.")

    return vectors


def cosine_similarity(query_vector, document_vectors):
    """Calculate cosine similarity."""
    query_norm = np.linalg.norm(query_vector)
    document_norms = np.linalg.norm(document_vectors, axis=1)

    if query_norm == 0 or np.any(document_norms == 0):
        raise ValueError("Cannot calculate similarity for zero vectors.")

    return (
        document_vectors @ query_vector
    ) / (document_norms * query_norm)


@lru_cache(maxsize=1)
def load_knowledge_base():
    """Load and embed project evidence once per process."""
    if not CONTEXT_PATH.exists():
        raise FileNotFoundError(
            f"Decision context not found: {CONTEXT_PATH}"
        )

    document = CONTEXT_PATH.read_text(encoding="utf-8")
    chunks = split_into_chunks(document)

    vectors = get_embeddings([
        chunk["text"] for chunk in chunks
    ])

    return chunks, vectors


def retrieve_evidence(question, top_k=TOP_K):
    """Hybrid semantic and lexical evidence retrieval."""
    chunks, document_vectors = load_knowledge_base()

    query_vector = get_embeddings([question])[0]

    semantic_scores = cosine_similarity(
        query_vector,
        document_vectors,
    )

    query_terms = tokenize(question)

    if not query_terms:
        raise ValueError("The question has no searchable terms.")

    ranked_results = []

    for index, chunk in enumerate(chunks):
        document_terms = tokenize(chunk["text"])

        overlap = len(query_terms & document_terms)
        lexical_score = overlap / len(query_terms)

        semantic_score = float(semantic_scores[index])

        relevance_score = (
            0.80 * semantic_score
            + 0.20 * lexical_score
        )

        ranked_results.append({
            "heading": chunk["heading"],
            "text": chunk["text"],
            "similarity": semantic_score,
            "relevance_score": relevance_score,
        })

    ranked_results.sort(
        key=lambda item: item["relevance_score"],
        reverse=True,
    )

    relevant_results = [
        item
        for item in ranked_results
        if item["relevance_score"] >= MIN_RELEVANCE_SCORE
    ]

    return relevant_results[:top_k]


def select_relevant_evidence(question, evidence):
    """
    Restrict evidence to the project section that matches
    the user's question.
    """
    q = question.lower()

    # Forecasting questions
    if any(term in q for term in [
        "forecast", "forecasting", "mae", "rmse",
        "mape", "holt-winters", "seasonal naive",
    ]):
        keywords = ["revenue forecasting"]

    # Profit optimization questions
    elif any(term in q for term in [
        "incremental profit", "net profit",
        "profit per", "profit optimization",
        "profitable", "email cost",
    ]):
        keywords = ["profit-based campaign optimization"]

    # Campaign policy / allocation questions
    elif any(term in q for term in [
        "campaign allocation", "allocation strategy",
        "uplift-based allocation", "random allocation",
        "incremental conversions", "campaign policy",
    ]) and not any(term in q for term in ["srm", "sample ratio"]):
        keywords = ["campaign policy evaluation"]

    # Conversion experiment and SRM questions
    elif any(term in q for term in [
        "srm", "sample ratio", "conversion rate",
        "pairwise conversion", "holm correction",
        "statistically significant differences",
    ]):
        keywords = ["campaign experimentation results"]

    # Uplift modeling questions
    elif any(term in q for term in [
        "uplift model", "qini", "t-learner",
        "s-learner", "x-learner",
    ]):
        keywords = ["uplift modeling"]

    # Purchase propensity questions
    elif any(term in q for term in [
        "purchase propensity", "propensity model",
        "purchase prediction",
    ]):
        keywords = ["purchase propensity"]

    # Churn / inactivity questions
    elif any(term in q for term in [
        "churn", "customer inactivity", "inactive customer",
    ]):
        keywords = ["customer inactivity prediction"]

    else:
        return evidence

    selected = [
        item for item in evidence
        if any(
            keyword in item["heading"].lower()
            for keyword in keywords
        )
    ]

    return selected


def generate_profit_answer():
    """
    Return the verified project profit results directly.

    This avoids the language model omitting confidence intervals
    or making unsupported statistical-significance claims.
    """
    return """**Direct Answer**

Random 30% allocation generated the highest estimated incremental profit: **£227.95 per 1,000 customers**, compared with £167.94 for uplift-based 30% allocation.

**Results**

| Metric | No Campaign | Random 30% | Uplift-Based 30% |
|---|---:|---:|---:|
| Contact rate | 0.00% | 30.00% | 29.99% |
| Revenue per customer | £0.6528 | £0.8837 | £0.8237 |
| Cost per customer | £0.0000 | £0.0030 | £0.0030 |
| Net profit per customer | £0.6528 | £0.8807 | £0.8207 |
| Incremental profit per 1,000 | £0.00 | £227.95 | £167.94 |

**Bootstrap Confidence Intervals**

| Strategy | Confidence Interval for Incremental Profit per 1,000 |
|---|---:|
| Random 30% allocation | £142.23 to £313.86 |
| Uplift-based 30% allocation | -£1.53 to £337.45 |

**Interpretation**

Random allocation produced the higher estimated incremental profit. However, the uplift-based policy's confidence interval includes zero. The evidence does not establish that uplift targeting is profitable or superior to random allocation.

**Limitations**

- The email cost assumption is illustrative: £0.01 per email, not verified.
- Evaluation used five-fold outer cross-fitting.
- Bootstrap intervals use 2,000 customer-level resamples and hold learned policy decisions fixed; they do not include model-retraining uncertainty.
- Results estimate policy value in the historical randomized experiment and do not guarantee future campaign profit.

[Source: 7. Profit-Based Campaign Optimization]
[Source: 7. Profit-Based Campaign Optimization > Finding]
[Source: 7. Profit-Based Campaign Optimization > Methodology and Limitations]"""


def generate_answer(question, evidence):
    """Generate a focused answer grounded in selected evidence."""

    if not evidence:
        return (
            "I could not retrieve sufficiently relevant evidence "
            "to answer this question. Please check the question "
            "against the project context."
        )

    q = question.lower()

    # Use the verified deterministic answer for profit questions.
    if (
        "incremental profit" in q
        or "profit optimization" in q
        or "net profit" in q
    ):
        return generate_profit_answer()

    evidence_text = "\n\n".join(
        (
            f"SOURCE: {item['heading']}\n"
            f"CONTENT:\n{item['text']}"
        )
        for item in evidence
    )

    system_prompt = """
You are the RetailIQ business decision assistant.

Answer the user's question using ONLY the supplied project evidence.

MANDATORY RULES:

1. Evidence accuracy
- Never invent numbers, statistics, p-values, confidence intervals,
  sample sizes, assumptions, or limitations.
- Every numerical claim must appear explicitly in the evidence
  or be a transparent arithmetic calculation from reported values.
- Do not infer exact p-values from a statement of statistical
  significance.
- If exact pairwise p-values are absent, do not create them.

2. Scope
- Answer only the question asked.
- Use only the supplied evidence relevant to that question.
- Do not introduce other campaign strategies, policy evaluations,
  profit results, or model results unless explicitly requested.
- Do not add an unavailable-information section unless the
  missing information was explicitly requested.

3. Statistical interpretation
- Distinguish statistical significance from practical importance.
- Report Holm-adjusted significance only as stated in the evidence.
- Do not invent individual pairwise p-values.
- Do not claim statistical superiority without supporting evidence.
- Overlapping confidence intervals do not prove equivalence.

4. Metrics
- Preserve all reported values, units, and precision.
- Do not confuse ROC-AUC, PR-AUC, precision, recall, or F1.
- For requested model comparisons, include every relevant model
  and every requested metric.
- Identify metric leaders separately.

5. Limitations
- Include only limitations explicitly stated in the relevant evidence.
- Do not invent methodological weaknesses.
- Do not import limitations from unrelated sections.

6. Citations
- Cite the exact source heading provided in the evidence.
- Use this format:
  [Source: exact heading]
- Never cite evidence by number.
- Never invent a source heading.

7. Response quality
- Start with a direct answer.
- Use a table when comparing multiple groups or models.
- Avoid duplicate tables and repeated explanations.
- Include only relevant limitations.
- Keep the answer concise but complete.

IMPORTANT:
If the evidence states that all pairwise comparisons are
statistically significant after Holm correction but does not
provide exact pairwise p-values, report the significance only.
Do not generate p-values such as <0.001.
"""

    payload = {
        "model": CHAT_MODEL,
        "stream": False,
        "options": {
            "temperature": 0,
        },
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\n"
                    f"PROJECT EVIDENCE:\n{evidence_text}\n\n"
                    "Answer only the requested question. "
                    "Use exact source headings for citations."
                ),
            },
        ],
    }

    response = requests.post(
        CHAT_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    answer = response.json().get(
        "message", {}
    ).get("content")

    if not answer:
        raise ValueError("Ollama returned an empty answer.")

    return answer


@app.post("/assistant/ask")
def ask_assistant(request: Question):
    try:
        retrieved_evidence = retrieve_evidence(
            request.question
        )

        relevant_evidence = select_relevant_evidence(
            request.question,
            retrieved_evidence,
        )

        answer = generate_answer(
            request.question,
            relevant_evidence,
        )

        return {
            "answer": answer,
            "retrieved_evidence": [
                {
                    "source": item["heading"],
                    "similarity": round(
                        item["similarity"], 4
                    ),
                    "relevance_score": round(
                        item["relevance_score"], 4
                    ),
                    "text": item["text"],
                }
                for item in relevant_evidence
            ],
            "retrieval_count": len(relevant_evidence),
            "embedding_model": EMBEDDING_MODEL,
            "generation_model": CHAT_MODEL,
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama connection failed: {exc}",
        )

    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"RAG processing failed: {exc}",
        )