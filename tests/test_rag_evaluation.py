
import requests
import pytest


API_URL = "http://localhost:8000/assistant/ask"
TIMEOUT = 240


def ask_assistant(question):
    """Send a question to the running RetailIQ RAG API."""
    response = requests.post(
        API_URL,
        json={"question": question},
        timeout=TIMEOUT,
    )

    response.raise_for_status()
    return response.json()


def normalize(text):
    """Normalize answer text for reliable assertions."""
    return " ".join(text.lower().split())


def check_evidence(result, expected_source):
    """Verify that the expected project evidence was retrieved."""
    sources = [
        item["source"].lower()
        for item in result["retrieved_evidence"]
    ]

    assert result["retrieval_count"] > 0
    assert any(
        expected_source.lower() in source
        for source in sources
    )


def test_campaign_conversion_and_srm():
    question = (
        "Compare the campaign conversion rates. "
        "Which differences are statistically significant, "
        "and was there evidence of sample ratio mismatch (SRM)?"
    )

    result = ask_assistant(question)
    answer = normalize(result["answer"])

    assert "1.253%" in answer
    assert "0.573%" in answer
    assert "0.884%" in answer
    assert "0.2025" in answer
    assert "0.9037" in answer
    assert "holm" in answer
    assert "statistically significant" in answer

    check_evidence(
        result,
        "3. Campaign Experimentation Results",
    )


def test_campaign_profit_optimization():
    question = (
        "Which campaign allocation strategy generated "
        "the highest estimated incremental profit? "
        "Explain the results and limitations."
    )

    result = ask_assistant(question)
    answer = normalize(result["answer"])

    assert "£227.95" in answer
    assert "£167.94" in answer
    assert "£142.23" in answer
    assert "£313.86" in answer
    assert "-£1.53" in answer
    assert "£337.45" in answer

    assert "five-fold" in answer
    assert "£0.01" in answer
    assert "2,000" in answer
    assert "model-retraining uncertainty" in answer

    assert (
        "does not establish that uplift targeting is profitable"
        in answer
    )

    check_evidence(
        result,
        "7. Profit-Based Campaign Optimization",
    )


def test_revenue_forecasting():
    question = (
        "Compare the forecasting models using MAE, RMSE, "
        "and MAPE. Which model performed best, and why?"
    )

    result = ask_assistant(question)
    answer = normalize(result["answer"])

    assert "£147,508.75" in answer
    assert "£255,044.70" in answer
    assert "21.53%" in answer

    assert "£274,734.11" in answer
    assert "£395,445.27" in answer
    assert "23.10%" in answer

    assert "seasonal naive" in answer
    assert "holt-winters" in answer

    check_evidence(
        result,
        "9. Revenue Forecasting",
    )
