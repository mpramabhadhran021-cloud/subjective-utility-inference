"""
analysis_helpers.py

Shared helper functions for the two-notebook project on statistical inference
of subjective preferences from choices under uncertainty.

This module contains:
    (1) a data-generating process (DGP) used to produce a SIMULATED dataset
        that stands in for real experimental data during development,
    (2) light data-quality corruption utilities, used to give the data-
        cleaning section of Notebook 1 something concrete to work with,
    (3) cleaning utilities,
    (4) an approximate certainty-equivalent estimator,
    (5) a clustered bootstrap helper for participant-level uncertainty.

IMPORTANT
---------
All data produced by `simulate_participants` / `simulate_choices` is
SIMULATED. It is used only because no real participant pool was available
for this exercise. Every place the data are used, they are labelled as
simulated. The underlying preference parameters (alpha, lam, gamma, k) are
never exposed to the "researcher" side of the analysis -- Notebook 2 only
ever sees choices, not these generating parameters -- so that the inference
exercise is honest: the notebooks recover *evidence about* preferences from
choices, not the parameters themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1. Experimental design
# --------------------------------------------------------------------------

def build_design(
    probabilities=(0.10, 0.25, 0.50, 0.75, 0.90),
    magnitude=100,
    certain_offsets=(-10, 0, 10),
    include_dominance_checks=True,
):
    """
    Build the trial-level experimental design (independent of participants).

    For every probability level and every domain (gain / loss), three
    certain amounts are placed around the expected value of the risky
    prospect (EV - 10, EV, EV + 10), following the bracketing logic in the
    project brief (cf. Section 11, "Experimental Questions").

    Two dominance-check trials are added (one per domain) where the certain
    option strictly dominates the risky option. A participant who fails a
    dominance check is not necessarily wrong, but persistent dominance
    violations are used later as one (imperfect) marker of inattentive
    responding.

    Returns
    -------
    pd.DataFrame with columns:
        trial_id, domain, probability, gain, loss, certain_amount, is_dominance_check
    """
    rows = []
    trial_id = 1
    for domain in ("gain", "loss"):
        sign = 1 if domain == "gain" else -1
        for p in probabilities:
            ev = p * magnitude
            for off in certain_offsets:
                certain = sign * max(0, min(magnitude, ev + off))
                rows.append({
                    "trial_id": trial_id,
                    "domain": domain,
                    "probability": p,
                    "gain": magnitude if domain == "gain" else 0,
                    "loss": magnitude if domain == "loss" else 0,
                    "certain_amount": certain,
                    "certain_offset": off,
                    "is_dominance_check": False,
                })
                trial_id += 1
        if include_dominance_checks:
            # Certain option strictly better than every possible risky outcome.
            # Gains: a certain amount above the best possible lottery outcome.
            # Losses: a guaranteed zero loss, which weakly beats every
            # lottery outcome (0 or -100) and strictly beats the bad state.
            dom_certain = (magnitude + 20) if domain == "gain" else 0
            rows.append({
                "trial_id": trial_id,
                "domain": domain,
                "probability": 0.5,
                "gain": magnitude if domain == "gain" else 0,
                "loss": magnitude if domain == "loss" else 0,
                "certain_amount": dom_certain,
                "certain_offset": np.nan,
                "is_dominance_check": True,
            })
            trial_id += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Participant-level preference parameters (SIMULATED, latent)
# --------------------------------------------------------------------------

def simulate_participants(n_participants=60, seed=1):
    """
    Draw latent, unobservable preference parameters for each simulated
    participant. These parameters are the "ground truth" used only to
    generate choices; the statistical analysis in Notebook 2 never has
    access to them, mirroring the fact that real researchers never observe
    true preference parameters either.

    alpha : curvature of the value function (diminishing sensitivity).
            alpha < 1 -> concave in gains / convex in losses (standard).
    lam   : loss-aversion multiplier applied to the loss branch.
    gamma : probability-weighting exponent (gamma < 1 -> overweight small
            probabilities, underweight large ones -- inverse-S weighting).
    k     : choice-sensitivity ("consistency") parameter for the logistic
            choice rule. Lower k -> noisier, less consistent choices.
    """
    rng = np.random.default_rng(seed)

    alpha = np.clip(rng.normal(0.78, 0.15, n_participants), 0.35, 1.15)
    lam = np.clip(rng.normal(1.8, 0.6, n_participants), 0.7, 3.5)
    gamma = np.clip(rng.normal(0.75, 0.15, n_participants), 0.40, 1.10)
    k = np.clip(rng.normal(0.16, 0.07, n_participants), 0.03, 0.40)

    age_group = rng.choice(["18-24", "25-34", "35-44", "45+"],
                            size=n_participants, p=[0.45, 0.30, 0.15, 0.10])
    education = rng.choice(["undergraduate", "postgraduate", "other"],
                            size=n_participants, p=[0.55, 0.35, 0.10])
    quant_background = rng.choice(["yes", "no"], size=n_participants, p=[0.4, 0.6])

    participant_id = [f"P{i+1:03d}" for i in range(n_participants)]

    return pd.DataFrame({
        "participant_id": participant_id,
        "age_group": age_group,
        "education": education,
        "quant_background": quant_background,
        # latent, not to be used downstream except for DGP validation:
        "_alpha": alpha, "_lam": lam, "_gamma": gamma, "_k": k,
    })


# --------------------------------------------------------------------------
# 3. Value function, probability weighting, choice simulation
# --------------------------------------------------------------------------

def value_fn(x, alpha, lam):
    """Simplified value function with diminishing sensitivity and loss aversion."""
    x = np.asarray(x, dtype=float)
    pos = np.power(np.abs(x), alpha)
    return np.where(x >= 0, pos, -lam * pos)


def prob_weight(p, gamma):
    """Simple one-parameter probability weighting function, w(p) = p**gamma."""
    return np.power(p, gamma)


def simulate_choices(design_df, participants_df, seed=2, noise_scale=0.6):
    """
    Simulate one choice per (participant, trial) using a probability-weighted,
    loss-averse value function combined with a logistic (softmax) choice rule.

    P(choose risky) = sigmoid( k_i * (V_risky - V_certain) / noise_scale )

    `noise_scale` rescales the value difference so that k, calibrated in
    simulate_participants, produces acceptance rates that vary sensibly
    across trials rather than collapsing to 0/1. It has no substantive
    interpretation; it is a convenience of the simulation, not a modelling
    claim about real decision-makers.
    """
    rng = np.random.default_rng(seed)
    records = []

    for _, ptn in participants_df.iterrows():
        alpha, lam, gamma, k = ptn["_alpha"], ptn["_lam"], ptn["_gamma"], ptn["_k"]
        order = design_df.sample(frac=1.0, random_state=rng.integers(1e9)).reset_index(drop=True)

        for trial_number, (_, tr) in enumerate(order.iterrows(), start=1):
            risky_outcome = tr["gain"] if tr["domain"] == "gain" else -tr["loss"]
            v_risky_outcome = value_fn(risky_outcome, alpha, lam)
            w = prob_weight(tr["probability"], gamma)
            V_risky = w * v_risky_outcome  # value_fn(0, ...) == 0, so the
                                            # complementary branch drops out
            V_certain = value_fn(tr["certain_amount"], alpha, lam)

            diff = (V_risky - V_certain) / noise_scale
            p_risky = 1.0 / (1.0 + np.exp(-k * diff))
            choice = "gamble" if rng.random() < p_risky else "certain"

            records.append({
                "participant_id": ptn["participant_id"],
                "trial_number": trial_number,
                "trial_id": tr["trial_id"],
                "domain": tr["domain"],
                "probability": tr["probability"],
                "gain": tr["gain"],
                "loss": tr["loss"],
                "certain_amount": tr["certain_amount"],
                "certain_offset": tr["certain_offset"],
                "is_dominance_check": tr["is_dominance_check"],
                "choice": choice,
            })

    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# 4. Synthetic data-quality issues (so Notebook 1's cleaning section is real)
# --------------------------------------------------------------------------

def corrupt_data(df, seed=3, missing_frac=0.012, duplicate_frac=0.01,
                  impossible_frac=0.006):
    """
    Inject a small, realistic amount of data-quality noise into an otherwise
    clean simulated dataset: missing responses, duplicated rows, and a few
    impossible values. This mimics the kinds of issues a real choice
    experiment would produce and gives the cleaning section (Notebook 1,
    Section 7) genuine work to do rather than a cosmetic exercise.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)

    # Missing responses
    miss_idx = rng.choice(n, size=int(n * missing_frac), replace=False)
    df.loc[miss_idx, "choice"] = np.nan

    # Duplicated rows (participant answered/was logged twice)
    dup_idx = rng.choice(n, size=int(n * duplicate_frac), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # Impossible values: corrupt probability or certain_amount entries
    imp_idx = rng.choice(len(df), size=int(n * impossible_frac), replace=False)
    for i, idx in enumerate(imp_idx):
        if i % 2 == 0:
            df.loc[idx, "probability"] = 1.4  # impossible probability
        else:
            df.loc[idx, "certain_amount"] = np.nan

    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def clean_data(df):
    """
    Clean a raw (possibly corrupted) choices dataframe.

    Steps (documented, non-silent):
        1. Drop exact duplicate rows.
        2. Drop rows with a missing `choice`.
        3. Drop rows with an impossible `probability` (outside (0, 1]).
        4. Drop rows with a non-finite or missing `certain_amount`.

    Returns
    -------
    (clean_df, report) where report is a dict with counts removed at each step.
    """
    report = {"n_raw": len(df)}

    d = df.drop_duplicates()
    report["duplicates_removed"] = len(df) - len(d)

    before = len(d)
    d = d[d["choice"].notna()]
    report["missing_choice_removed"] = before - len(d)

    before = len(d)
    d = d[(d["probability"] > 0) & (d["probability"] <= 1)]
    report["impossible_probability_removed"] = before - len(d)

    before = len(d)
    d = d[np.isfinite(d["certain_amount"])]
    report["impossible_certain_amount_removed"] = before - len(d)

    d = d.reset_index(drop=True)
    report["n_clean"] = len(d)
    return d, report


# --------------------------------------------------------------------------
# 5. Approximate certainty equivalents
# --------------------------------------------------------------------------

def approximate_certainty_equivalent(sub_df):
    """
    Given the rows for ONE participant x ONE probability x ONE domain
    (i.e. the 3 certain-amount levels bracketing the EV), return an
    approximate certainty equivalent (CE) using the switching point between
    "gamble" and "certain" choices as certain_amount increases in
    magnitude.

    Returns np.nan if:
      - fewer than 3 observations survive cleaning for this triple (one
        record may have been removed in Notebook 1, Section 7). Computing a
        midpoint from only 1-2 remaining brackets would silently understate
        how much evidence the estimate is actually based on, and -- worse --
        would sometimes coincide with the EV by construction (e.g. if the
        middle bracket is the one missing, the midpoint of the two outer
        brackets equals the EV exactly, which looks like a meaningful
        "risk-neutral" result but is really just an artefact of a dropped
        row). Excluding incomplete triples avoids that.
      - choices are not monotonic (no clean switch point) -- deliberately
        conservative: an inconsistent participant does not get a fabricated CE.
    """
    if len(sub_df) != 3:
        return np.nan  # incomplete triple after cleaning -- do not estimate
    d = sub_df.sort_values("certain_amount")  # ascending = worst-to-best in BOTH
                                               # domains (e.g. -60 < -50 < -40 is
                                               # worst-to-best for a loss; sorting
                                               # by |certain_amount| would reverse
                                               # the loss-domain ordering)
    amounts = d["certain_amount"].values
    choices = (d["choice"] == "gamble").values

    if choices.all():
        return np.nan  # always preferred gamble even at the most generous certain option: unbounded CE, cannot pin down
    if not choices.any():
        return np.nan  # never took the gamble: CE below range, cannot pin down

    # monotonic non-increasing preference for the gamble as |certain| rises?
    if not np.all(np.diff(choices.astype(int)) <= 0):
        return np.nan  # non-monotonic switching -> flagged, not estimated

    switch_i = np.argmax(~choices)  # first index where certain is chosen
    return (amounts[switch_i - 1] + amounts[switch_i]) / 2.0 if switch_i > 0 else np.nan


def classify_switching(sub_df):
    """
    Classify the choice pattern for ONE participant x ONE probability x ONE
    domain (3 certain-amount levels bracketing the EV) into one of:

        'incomplete'      -- fewer than 3 observations survived cleaning for
                             this triple (e.g. one record was removed as
                             missing or impossible in Notebook 1, Section 7).
                             Returned rather than silently classified from
                             1-2 points, which would rest on less evidence
                             than the label would imply.
        'always_gamble'   -- risky option chosen at every certain-amount level
        'never_gamble'    -- certain option chosen at every level
        'single_switch'   -- exactly one switch from gamble to certain as the
                             certain amount becomes objectively more generous
                             (a single-crossing pattern, the behaviour a
                             stable underlying preference would produce)
        'non_monotonic'   -- more than one switch (a "reversal"): the
                             participant prefers the gamble at a WORSE
                             certain amount than one at which they preferred
                             the certain option -- internally inconsistent
                             given only these three data points
    """
    if len(sub_df) != 3:
        return "incomplete"
    d = sub_df.sort_values("certain_amount")  # ascending = worst-to-best in
                                               # both domains -- see note in
                                               # approximate_certainty_equivalent
    choices = (d["choice"] == "gamble").values
    if choices.all():
        return "always_gamble"
    if not choices.any():
        return "never_gamble"
    if np.all(np.diff(choices.astype(int)) <= 0):
        return "single_switch"
    return "non_monotonic"


# --------------------------------------------------------------------------
# 6. Clustered (participant-level) bootstrap
# --------------------------------------------------------------------------

def cluster_bootstrap_ci(df, participant_col, statistic_fn, n_boot=2000,
                          seed=4, alpha=0.05):
    """
    Percentile bootstrap that resamples PARTICIPANTS (clusters), not
    individual trials, since trials from the same participant are not
    independent. `statistic_fn` takes a dataframe and returns a scalar.
    """
    rng = np.random.default_rng(seed)
    groups = {p: g for p, g in df.groupby(participant_col)}
    participants = np.array(list(groups.keys()))
    n = len(participants)
    stats = np.empty(n_boot)

    for b in range(n_boot):
        sampled = rng.choice(participants, size=n, replace=True)
        boot_df = pd.concat([groups[p] for p in sampled], ignore_index=True)
        stats[b] = statistic_fn(boot_df)

    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(stats)), float(lo), float(hi)

# --------------------------------------------------------------------------
# 7. Structural choice model
# --------------------------------------------------------------------------

def structural_choice_probability(params, probability, certain_amount, risky_outcome):
    """Return P(choose the risky option) for the structural model."""
    alpha, lam, gamma, beta = params
    risky_value = prob_weight(probability, gamma) * value_fn(risky_outcome, alpha, lam)
    certain_value = value_fn(certain_amount, alpha, lam)
    diff = beta * (risky_value - certain_value)
    return 1.0 / (1.0 + np.exp(-diff))


def fit_structural_model(df, x0=(0.8, 1.8, 0.75, 0.25)):
    """Fit the structural model by maximum likelihood."""
    from scipy.optimize import minimize

    y = df["gambled"].to_numpy(dtype=float)
    p = df["probability"].to_numpy(dtype=float)
    certain = df["certain_amount"].to_numpy(dtype=float)
    risky = df["risky_outcome"].to_numpy(dtype=float)

    def nll(params):
        prob = structural_choice_probability(params, p, certain, risky)
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob))

    bounds = [(0.05, 2.0), (0.2, 5.0), (0.2, 2.0), (0.001, 5.0)]
    result = minimize(nll, x0=x0, bounds=bounds, method="L-BFGS-B")
    result.nll = float(result.fun)
    result.aic = 2 * len(result.x) + 2 * result.fun
    result.bic = len(result.x) * np.log(len(df)) + 2 * result.fun
    return result
