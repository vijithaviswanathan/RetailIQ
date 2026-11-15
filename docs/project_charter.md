
# RetailIQ — Project Charter

## 1. Project Vision

Build a retail decision intelligence platform that combines
experimentation, statistical inference, causal machine learning,
predictive analytics, and AI-assisted reporting.

## 2. Primary Business Question

How can a retail business identify which marketing interventions
create incremental customer value and allocate its marketing
resources more effectively?

## 3. Key Research Questions

1. Do marketing campaigns increase conversion rates?
2. Are observed treatment effects statistically significant?
3. What is the incremental revenue attributable to treatment?
4. Which customers are most likely to benefit from treatment?
5. Can customer inactivity be predicted?
6. Can retail demand be forecast accurately?
7. Which campaign allocation strategy maximizes incremental profit?

## 4. Main Workstreams

- Data engineering and SQL analytics
- Exploratory data analysis
- Statistical inference
- Randomized experimentation
- Causal inference and uplift modeling
- Predictive machine learning
- Forecasting
- Business optimization
- AI-assisted experiment reporting
- Application development and monitoring

## 5. Data Strategy

Hillstrom Email Marketing will be used for historical randomized
campaign analysis.

UCI Online Retail II will be used for transactional retail analytics.

The datasets will not be merged as if they describe the same customers.

## 6. Methodological Standards

- Define hypotheses before interpreting results.
- Report effect sizes and confidence intervals.
- Check experiment assumptions and data quality.
- Separate randomized evidence from observational analysis.
- Prevent temporal and target leakage.
- Use appropriate train, validation, and test strategies.
- Document limitations and reproducibility.

## 7. Expected Deliverables

- Reproducible data pipelines
- SQL database and analytical queries
- Statistical experiment analysis
- Causal inference and uplift models
- Predictive ML models
- Forecasting and optimization analysis
- Interactive dashboard
- API and AI assistant
- Automated tests
- Technical documentation
- Final project demonstration

## 8. Project Boundaries

This project is a portfolio and research implementation.
It is not a claim of running a live production retail experiment.

Any simulated experiments will be explicitly labeled as simulations.
