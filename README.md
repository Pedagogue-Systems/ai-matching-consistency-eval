# Matching Consistency Benchmark

AI Matching Consistency Eval is a benchmark project by Pedagogue Systems exploring how different AI/ML models rank resumes against job descriptions. It investigates model agreement, scoring divergence, and the implications for fairness and explainability in staffing technology.

This project evaluates how consistent different AI/ML models are when matching resumes to job postings. It explores whether models agree on top candidates and investigates the implications for trust, validation, and responsible AI use in staffing.

## Objectives

- Compare AI models (e.g., OpenAI, HuggingFace, Unsloth) on identical job/resume inputs
- Measure ranking consistency and score divergence
- Visualize and interpret where models agree or disagree
- Provide a reproducible benchmark aligned with responsible AI practices

## Tools

- Python
- HuggingFace
- Scikit-learn
- Streamlit

## Structure

- `/data`: Sample resumes and job descriptions
- `/src`: Matching logic and utilities
- `/notebooks`: Jupyter notebook for analysis
- `/streamlit_app`: Demo app
- `requirements.txt`: Dependencies

## Getting Started

```bash
git clone https://github.com/Pedagogue-Systems/ai-matching-consistency-eval.git
cd ai-matching-consistency-eval
pip install -r requirements.txt
```

## Contributions

This project is part of an internship with Pedagogue Systems and reflects our commitment to ethical, explainable AI in workforce technology.
