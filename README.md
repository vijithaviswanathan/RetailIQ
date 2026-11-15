
# RetailIQ
## Retail Experimentation, Causal ML & AI Decision Intelligence Platform

RetailIQ is an end-to-end data science project that combines retail analytics, randomized campaign evaluation, causal machine learning, predictive modeling, forecasting, business optimization, and a locally hosted AI decision assistant.

## Business Problem

Retail businesses need to understand whether marketing campaigns generate incremental customer value, which customers may respond to interventions, and how marketing resources should be allocated.

RetailIQ uses statistical analysis and machine learning to investigate these questions and translate analytical findings into business recommendations.

## Key Features

- **Retail Analytics:** PostgreSQL-based data processing, data quality investigations, revenue analysis, and customer segmentation.
- **Experimentation:** Hypothesis testing, confidence intervals, sample-size calculations, statistical power, and sample ratio mismatch checks.
- **Causal Machine Learning:** S-Learner and T-Learner uplift modeling with Qini-based evaluation.
- **Predictive Modeling:** Customer purchase propensity and inactivity prediction.
- **Forecasting:** Monthly revenue forecasting with baseline comparisons.
- **Business Optimization:** Cross-fitted campaign policy evaluation and estimated incremental profit.
- **AI Decision Assistant:** Local LLM-powered assistant using Ollama and project decision context.
- **Interactive Dashboard:** Streamlit dashboard integrating analytical results and the AI assistant.
- **API:** FastAPI endpoint for asking questions about project findings.

## Datasets

1. **Hillstrom Email Marketing Dataset:** Used for randomized campaign analysis, conversion evaluation, uplift modeling, and campaign policy assessment.
2. **UCI Online Retail II:** Used for retail transaction analytics, customer segmentation, predictive modeling, and revenue forecasting.

The datasets are analyzed separately because they have different structures and no shared customer identifiers.

## Technology Stack

- Python, SQL, PostgreSQL
- Pandas, NumPy, SciPy, Statsmodels
- Scikit-learn
- SQLAlchemy
- FastAPI, Uvicorn
- Streamlit, Plotly
- Ollama (local LLM)
- Git and GitHub

## Project Structure

```text
RetailIQ/
├── docs/
├── notebooks/
├── reports/
├── src/retailiq/
│   ├── analytics/
│   ├── api/
│   ├── causal_ml/
│   ├── data/
│   ├── evaluation/
│   ├── experimentation/
│   ├── models/
│   └── dashboard.py
├── tests/
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RetailIQ
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure PostgreSQL

Create a PostgreSQL database named `retailiq`.

Configure the database connection using the project's database configuration.

### 5. Prepare datasets

Download the Hillstrom Email Marketing and UCI Online Retail II datasets separately.

Place the source files in the expected local data directory according to the project's loading scripts. Raw datasets are excluded from Git.

### 6. Start the dashboard

```bash
streamlit run src/retailiq/dashboard.py
```

### 7. Start the AI assistant API

In a separate terminal:

```bash
source .venv/bin/activate
ollama serve
```

Ensure the configured Ollama model is available locally, then run:

```bash
uvicorn retailiq.api.assistant:app --reload --app-dir src
```

API documentation:

`http://127.0.0.1:8000/docs`

Dashboard:

`http://localhost:8501`

## Research and Evaluation Principles

- Distinguish correlation from causation.
- Document statistical assumptions.
- Avoid data leakage.
- Use time-aware validation where appropriate.
- Report uncertainty and limitations.
- Distinguish observed outcomes from model estimates.
- Avoid overstating business impact or model performance.

## Limitations

- The Hillstrom dataset does not contain customer identifiers.
- Historical experimental findings may not generalize to future campaigns.
- Campaign cost assumptions are illustrative.
- Predictive and uplift models require further validation before production use.
- The AI assistant is grounded in project documentation and is not a substitute for independent analytical verification.

## Author

Individual Data Science Portfolio Project.
