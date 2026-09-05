# Data-Driven Retail Profit Optimization

A causal decision pipeline that goes beyond forecasting: it **predicts** demand,
**optimises** price and inventory decisions, **recommends** actions with an
explanation, and then **measures** the profit impact — grading every claim by the
strength of the evidence behind it.

Built on the dunnhumby *Breakfast at the Frat* retail panel (156 weeks, 77 stores,
58 products), with a schema-matched synthetic replica carrying planted ground truth
so the estimators can be validated against a known answer.

## Live dashboard

The interactive dashboard is deployed on Streamlit Community Cloud.
It reads the pre-computed results in `outputs/`, so no raw data is needed to run it.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Running the full pipeline

The dashboard needs three packages. The 11-phase pipeline needs more:

```bash
pip install -r requirements-dev.txt
```

The raw dunnhumby workbook is **not** in this repository — it is licensed and
too large for GitHub. Place it under `data/raw/frat/` and set `DATA_SOURCE`
in `src/config.py`, or run `src/make_synthetic_frat.py` to generate the replica.

Then run the phases in order:

```bash
python src/make_synthetic_frat.py   # optional: synthetic replica
python src/data_loader.py           # canonicalise + weekly panel
python src/elasticity.py            # naive / FE / IV / shrunk elasticities
python src/uplift.py                # promotion uplift via DiD
python src/forecast.py              # gradient-boosted demand forecast
python src/pricing.py               # Lerner optimum under guardrails
python src/markdown.py              # finite-horizon markdown MDP
python src/inventory.py             # forecast-driven replenishment
python src/explain.py               # SHAP + exact formula explanations
python src/measure.py               # profit ledger + evidence grading
streamlit run app.py                # dashboard
```

## What is in here

| Path | Purpose |
|---|---|
| `src/` | The 11-phase pipeline, one module per phase |
| `outputs/` | Committed results — what the dashboard reads |
| `app.py` | Streamlit dashboard (6 tabs) |
| `paper/` | IEEE conference paper (LaTeX, IEEEtran) and figures |

## Method notes

- **Elasticity** — price is endogenous, so ordinary regression is biased. A
  Hausman instrument (leave-one-out mean log price of the same UPC in other
  stores in the same week) identifies the demand curve; estimates are then
  shrunk toward the category median by empirical Bayes.
- **Promotion uplift** — difference-in-differences with two-way fixed effects,
  checked with a parallel-trends placebo.
- **Forecasting** — gradient boosting on log units with a Duan smearing
  retransformation correction; scored with WAPE on a strictly temporal split.
- **Pricing** — the Lerner condition `p* = c·ε/(1+ε)`, bounded by guardrails
  (max move, margin floor, competitive band).
- **Markdown** — backward induction over a finite horizon; cost is sunk, so the
  objective is cash recovered rather than margin.
- **Evidence grading** — every headline number is labelled MEASURED, REPLAYED,
  OBSERVATIONAL or MODELLED. The gap between the modelled pricing gain and the
  inconclusive observational estimate is reported, not hidden.

## Author

Varshni K — Vellore Institute of Technology
