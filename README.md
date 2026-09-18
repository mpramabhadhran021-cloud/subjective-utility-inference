# Subjective Utility Inference

## What is this project?

This project studies whether choices between a certain payment and a gamble can provide information about risk preferences.

The basic idea is simple: we cannot directly observe someone's utility function, but we can observe the choices they make.

## Research Questions

1. Do people make different choices when facing the same gamble?
2. Can choices provide evidence about risk preferences?
3. How much do preferences differ across people?
4. Do choices differ between gains and losses?
5. Can a simple statistical model explain the choices?

## Data

The data are simulated rather than collected from real participants. I generate choices for 60 simulated participants using a simple model with risk preferences, probability weighting, loss aversion, and choice noise.

The results are therefore a demonstration of a statistical method, not evidence about real people.

## Method

I use exploratory analysis, certainty-equivalent calculations, participant-level bootstrap intervals, logistic regression, a random-effects model, model comparison, maximum-likelihood estimation, and basic robustness checks.

## Main Findings

The simulated participants show differences in their choices, and the models recover evidence of participant-level heterogeneity.

The project demonstrates how preference information can be estimated from choice data.

## Repository

notebooks/01_subjective_utility_experiment.ipynb — experiment and data
notebooks/02_statistical_inference.ipynb — statistical analysis
data/ — simulated data
figures/ — plots
src/ — helper code

## How to Run

pip install -r requirements.txt
jupyter notebook notebooks/01_subjective_utility_experiment.ipynb

Run Notebook 1 before Notebook 2.

## Limitations

The biggest limitation is that the data are simulated. The model is also only one possible description of choice behaviour, so different assumptions could give different estimates.