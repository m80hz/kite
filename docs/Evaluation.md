# Evaluation

We report metrics per task and question type.

## MCQ tasks
- Failure Detection, Failure Identification, Failure Locating
- Scoring: normalized string containment (case/space normalized). We compare predicted vs reference text for containment.

## Descriptive QA (non‑LLM)
For free‑form questions (Task identification, Task planning, Failure explanation, High‑level correction, Low‑level correction), we compute per‑sample metrics and average per type:

- Exact Match (normalized)
- Token F1 (SQuAD‑style unigram overlap)
- ROUGE‑L (F1) — requires `rouge-score`
- BLEU (sentence‑level) — requires `sacrebleu`
- chrF — requires `sacrebleu`
- SBERT cosine similarity in [0,1] (optional) — requires `sentence-transformers`

Install optional packages to enable all:

```
pip install rouge-score sacrebleu sentence-transformers
```

Outputs:
- Per‑split: `<out_dir>/stats_data.json` contains MCQ results and descriptive metric averages.
- Per‑type predictions: `<out_dir>/{task}_{question_type}_results.json` with per‑sample `metrics` for descriptive QAs.
- Merged across splits: `<out_root>/results_merged.json` weighted averages; nested metric dicts are merged field‑wise.

Notes:
- Metrics scale to [0,1]; BLEU/chrF values are normalized by 100.
- If optional deps aren’t installed, missing metrics are `null` and excluded from averages.
