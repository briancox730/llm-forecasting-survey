# LLM Forecasting Survey

Can a fine-tuned language model produce calibrated probabilities on resolvable yes/no questions, well enough to be worth money in an event market like Kalshi? This repo is a survey of the evidence plus an experiment design that answers the question cheaply, before committing build time.

It is docs only. No model has been trained. The point of the repo is the decision, not a deliverable: read the field first, define the accuracy gate that would justify a build, and only build if a two-day probe clears it.

## Why this question

"Retrain an LLM to trade" almost always means one of two very different things, and they get conflated constantly.

- **Trading agents** wrap an off-the-shelf model in a news-and-research scaffold that emits BUY/HOLD/SELL. Results are mostly backtests, usually fragile, and rarely beat a cheap baseline out of sample after costs.
- **Forecasting models** are RL-fine-tuned to emit a calibrated probability on a resolvable yes/no question, scored against the realized outcome with a proper scoring rule such as Brier or log loss. This branch is real, reproducible, and measurably improves with training.

The forecasting branch is the interesting one. It is the only place where "we trained a model and it got better at the thing prediction markets price" is cleanly true, so it is the only branch this repo pursues. It works on judgmental questions that have textual evidence behind them: economic releases, central-bank decisions, policy, and sports. It does not work on a fifteen-minute crypto price contract, which carries almost no text per window, so a language model has nothing to condition on.

## What the survey found

- The cleanest existence proof is the Mantic and Thinking Machines forecasting model. RL fine-tuning took a mediocre open model to near-frontier forecasting accuracy on unseen binary questions, and made it ensemble-valuable because its errors decorrelate from frontier models.
- Frontier LLMs today forecast at about the level of an engaged human crowd, below professional superforecasters. Fine-tuning closes the gap to frontier with a small open model. It does not yet beat superforecasters.
- Forecasting skill is highly category-dependent. A model can be well-calibrated in one domain and near-random in another, so any deployment has to find the calibrated slice and ignore the rest.
- Cost of entry is low. LoRA-scale fine-tunes run for a few hundred dollars. The bottleneck is data and a leakage-free calibration harness, not compute.

## How to read this repo

- [`docs/survey.md`](docs/survey.md) is the literature survey. It covers roughly thirty papers, repos, and posts, sorts them into the trading branch and the forecasting branch, and grades each on how much its training actually mattered and how much its results hold up.
- [`docs/experiment-design.md`](docs/experiment-design.md) is a phased build plan with a go/no-go gate at every phase. Phase 0 is a zero-training probe that can kill the project in two days. Nothing gets fine-tuned, and no money is risked, until a cheaper experiment has earned the next step.

Read the survey first, then the experiment design.

## Layout

```
README.md                    this file
docs/survey.md               the graded literature survey
docs/experiment-design.md    the phased, calibration-gated build plan
LICENSE                      MIT
```
