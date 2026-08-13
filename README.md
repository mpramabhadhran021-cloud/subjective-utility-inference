# Statistical Inference of Subjective Preferences from Choices Under Uncertainty

### One-sentence description

This project asks whether observed choices in a simple gamble-vs-certain-amount task
provide statistical evidence about individuals' subjective preferences toward risk,
without ever claiming to observe those preferences directly.

## Motivation

Objective outcomes — a rupee amount, a probability, a win or a loss — can be measured
and controlled precisely. The value an individual places on those outcomes cannot: two
people facing an identical 50/50 gamble may make different choices because they assign
different subjective value to the possible outcomes, not because they disagree about the
arithmetic. This project treats that gap as the object of study. It does not attempt to
recover anyone's "true" utility function; it asks what repeated, controlled choices can
and cannot tell us, statistically, about systematic differences in preference.

## Research Questions

1. Do individuals make systematically different choices when faced with the same
   objective outcomes?
2. Can observable choices reveal evidence of different attitudes toward risk?
3. How heterogeneous are preferences across individuals?
4. How does probability affect choices?
5. How do the size and direction of outcomes affect choices?
6. Do individuals respond differently to gains and losses?
7. How internally consistent are individual choices?
8. Can a relatively simple statistical model explain observed choices?
9. What can and cannot be inferred about subjective utility from choice data?

## Experimental Design

A compact binary-choice task: on each trial, participants choose between a guaranteed
("certain") monetary amount and a lottery paying a fixed non-zero amount with a stated
probability, ₹0 otherwise. Probability (0.10–0.90, 5 levels), certain amount (3 levels
bracketing the lottery's expected value), and domain (gain vs. loss) are crossed
systematically, giving 30 substantive trials plus 2 dominance-check trials per
participant (32 total). Full details, including randomisation and the reasoning behind
each design choice, are in `notebooks/01_subjective_utility_experiment.ipynb`, Section 5.

## Data

**The dataset used throughout this project is simulated**, not collected from real
participants — no data-collection round was run for this exercise. The simulation
(`src/analysis_helpers.py`) draws latent, unobservable preference parameters (risk
curvature, loss aversion, probability weighting, choice consistency) for 60 simulated
participants and generates choices from a probability-weighted, loss-averse value
function passed through a logistic choice rule. These latent parameters are never
exposed to the analysis side of the project — Notebook 2 only ever sees choices, exactly
as a real researcher would see only choices and not preferences. A small amount of
realistic data-quality noise (missing responses, duplicate rows, a few impossible field
values) is injected into the raw export so that the cleaning step is doing genuine work.

Every result in this project should be read as a demonstration of the statistical
approach applied to data with realistic properties — not as a finding about real human
decision-makers.

```
data/raw/participants_simulated.csv     # non-identifying background variables only
data/raw/choices_raw_simulated.csv      # simulated choices + injected data-quality noise
data/processed/choices_clean.csv        # cleaned, analysis-ready dataset
```

## Methods

- Exploratory analysis of acceptance rates by probability, certain amount, and domain
  (Notebook 1).
- Individual-level consistency checks (single-switch vs. non-monotonic patterns) and
  approximate certainty-equivalent estimation (Notebook 2, Sections 2–3).
- Cluster (participant-level) bootstrap confidence intervals and paired hypothesis tests
  for domain differences (Notebook 2, Section 4).
- A clustered-standard-error logistic regression choice model, interpreted via predicted
  probabilities rather than coefficients alone (Notebook 2, Section 5).
- A simulation-based overdispersion test for participant heterogeneity, cross-checked
  against a random-intercept (hierarchical) logistic model fit by variational Bayes
  (Notebook 2, Sections 6–7).
- Grouped cross-validation and AIC/BIC model comparison between a purely
  monetary-characteristics benchmark and a model incorporating gain/loss domain
  structure (Notebook 2, Section 9).
- A structural, theory-based choice model — the value-function and probability-weighting
  mechanism from Notebook 1, Section 3 — estimated by maximum likelihood directly from
  choices and compared against the reduced-form models on fit and predictive performance
  (Notebook 2, Section 10).
- Calibration diagnostics and two targeted robustness checks (Notebook 2, Sections
  11–12).

## Results

(Summarised from the executed notebooks; see `notebooks/02_statistical_inference.ipynb`
for full detail, uncertainty estimates, and caveats.)

- Gamble acceptance responds systematically to how the certain amount compares to the
  lottery's expected value, in the direction a stable preference would predict, and the
  effect is somewhat stronger in the loss domain than the gain domain.
- Gain-domain acceptance (54.7%) was higher than loss-domain acceptance (45.3%); the gap
  was significant by both a paired t-test and a Wilcoxon signed-rank test. The
  gain-domain certainty-equivalent distribution was close to a 50/50 split between
  risk-averse- and risk-seeking-consistent, while the loss domain leaned toward the
  risk-averse-consistent side — the opposite pattern from the textbook "reflection
  effect," and a reminder to check a theoretical prediction against the data rather than
  assume it.
- A formal overdispersion test found participant-level acceptance rates more dispersed
  than 15-trial sampling noise alone would predict (simulated one-sided p = 0.001) —
  genuine evidence of heterogeneity, not just a spread-out-looking histogram. A
  random-intercept logistic model reaches the same conclusion independently, estimating
  a non-trivial participant-level variance component (SD ≈ 0.14 log-odds units) after
  conditioning on probability, generosity, and domain.
- A structural model built from the theoretical value-function and probability-weighting
  mechanism — not a flexible reduced-form specification — matches the best reduced-form
  model's cross-validated accuracy exactly (69.2%) and slightly improves on its AIC/BIC,
  using the same number of parameters and none chosen after looking at the data;
  cross-validated log-loss favours the reduced-form model by a small margin, so the two
  criteria do not agree on every point. Fitted curvature and loss-aversion parameters are
  broadly in line with published estimates, but are pooled across a population already
  shown to be heterogeneous.
- In a joint logistic model, probability's own coefficient was not statistically
  distinguishable from zero once the certain option's EV-relative generosity and the
  gain/loss domain were included — a caution against reading too much into the
  univariate probability pattern.
- Adding gain/loss domain structure to the choice model improved out-of-sample
  cross-validated accuracy (66.6% → 69.2%) and was favoured by both AIC and BIC over a
  benchmark using only the certain option's generosity — a real, though not dramatic,
  improvement; cross-validated log-loss moved only slightly.
- About half of individual choice triples showed a single clean switch and roughly 79%
  of determinate triples were internally consistent with a single-threshold rule; the
  rest showed reversals, treated as evidence of measurement noise rather than of an
  actually non-monotonic preference.

## Limitations

The central limitation is that the data are simulated rather than real. Beyond that:
modest sample size for individual-level (as opposed to population-average) inference;
hypothetical/simulated rather than incentivised choices; gain and loss blocks were not
order-randomised relative to each other, so domain differences are confounded with
possible order effects; a single outcome magnitude (₹100) throughout; and every
preference-related quantity reported (certainty equivalents, model coefficients) is a
statistical summary filtered through modelling assumptions, not a direct measurement of
anyone's utility function. See Notebook 2, Section 12 for the full discussion.

## Reproducibility

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/01_subjective_utility_experiment.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_statistical_inference.ipynb
```

Notebook 1 must be run before Notebook 2 (it writes `data/processed/choices_clean.csv`).
All random draws are seeded, so re-running reproduces the dataset, figures, and results
in this repository exactly.

```
project/
├── README.md
├── notebooks/
│   ├── 01_subjective_utility_experiment.ipynb
│   └── 02_statistical_inference.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── figures/
├── src/
│   └── analysis_helpers.py
├── requirements.txt
└── LICENSE
```
