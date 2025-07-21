# AI Matching Consistency Validation Framework

This document defines the evaluation approach for consistency in AI-driven talent matching systems. It supports the broader goal of increasing reliability, transparency, and trust in autonomous staffing workflows, as outlined in the *AI in Staffing* article.

## Objective

Ensure that candidate–job matching scores remain consistent when inputs are semantically similar, while allowing for explainable and justifiable variance. This helps prevent noise, bias, or instability in recommendation systems.

---

## Key Evaluation Principles

1. **Consistency Across Similar Inputs**  
   - Equivalent or near-equivalent resumes should yield comparable match scores for the same job.
   - Equivalent jobs should yield comparable scores for the same resume.

2. **Explainable Variance**  
   - Score differences should be traceable to clear input distinctions (e.g., added certification, removed experience).

3. **Tolerance Thresholds**  
   - Acceptable variance levels (e.g., <5% change for minor edits).
   - Hard flags for significant divergence (>15% without clear explanation).

---

## Evaluation Methods

### 1. Paired Input Testing
Create controlled variations of candidate profiles and job postings:
- e.g., Resume A vs. Resume A' (with 1-line edit)
- e.g., Job A vs. Job A' (with 1-title tweak)

Run match scoring and compare outputs.

### 2. Similarity Clustering
Group profiles or jobs by semantic similarity (e.g., using sentence embeddings).
- Expectation: Low variance within clusters
- Flag: Outliers or scoring flips (e.g., top match drops to bottom)

### 3. Ranking Stability
Across multiple runs or variations, track changes in:
- Score deltas
- Relative rankings
- Final match/no-match threshold

---

## Metrics

| Metric                     | Description                                             |
|---------------------------|---------------------------------------------------------|
| Score Delta (%)           | Absolute % change between baseline and variant          |
| Rank Stability Index      | Position change in top-N matches                        |
| Variance within Cluster   | Standard deviation of scores in similar profiles/jobs   |
| False Flip Rate           | Cases where outcome flips (match ↔ no match)            |

---

## Output Format

All results should be exportable in the following format:

```json
{
  "input_pair": ["resume_A", "resume_A_prime"],
  "job": "job_123",
  "score_baseline": 0.82,
  "score_variant": 0.79,
  "delta": -0.03,
  "flag": "within_threshold"
}
