# Matching Consistency Benchmark

This project evaluates how consistent different AI/ML models are when matching resumes to job postings. It explores whether models agree on top candidates and investigates the implications for trust, validation, and responsible AI use in staffing.

## 🔍 Objectives

- Compare AI models (e.g., OpenAI, HuggingFace, Unsloth) on identical job/resume inputs
- Measure ranking consistency and score divergence
- Visualize and interpret where models agree or disagree
- Provide a reproducible benchmark aligned with responsible AI practices

## 🧰 Tools

- Python 3.10+
- LangChain
- HuggingFace Transformers
- Unsloth
- Scikit-learn or FAISS
- Streamlit (for dashboard)

## 🗂 Structure

- `/data`: Sample resumes and job descriptions
- `/src`: Matching logic and utilities
- `/notebooks`: Jupyter notebook for analysis
- `/streamlit_app`: Optional demo app
- `requirements.txt`: Dependencies

## 🚀 Getting Started

```bash
git clone https://github.com/Pedagogue-Systems/matching-consistency-benchmark.git
cd matching-consistency-benchmark
pip install -r requirements.txt
```

## 🤝 Contributions

This project is part of an internship with Pedagogue Systems and reflects our commitment to ethical, explainable AI in workforce technology.
