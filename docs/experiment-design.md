# Experiment design: an LLM forecaster for a Kalshi-style event market

Scoped from the survey. The only branch worth building is branch 4: an LLM that outputs calibrated probabilities on resolvable yes/no questions, applied to longer-horizon event markets (econ data prints, central-bank decisions, political and legislative outcomes, sports, "will X happen by date Y"). Not short-horizon crypto. That market has almost no text per window, so a language model has nothing to condition on.

Design principle: every phase is a gate. Nothing gets fine-tuned, and no real money is risked, until a cheaper experiment has earned the next step. You need a calibration harness first: Brier, log loss, a reliability curve, and a comparison against the market mid at the decision timestamp. Build it or reuse one, and do not skip it. Most of the risk in this project is in the measurement, not the model.

---

## Phase 0: the "should we even build this" probe (no code beyond data pulls; 1 to 2 days)

Goal: find out whether there is a slice of event markets where three things hold at once. First, textual evidence is available. Second, the market is liquid enough that a small edge survives fees. Third, frontier LLMs are already roughly calibrated. If any of the three fails, stop here and write a one-paragraph post-mortem.

1. **Inventory the event markets.** Pull the venue's market catalog through its API. Bucket by category (econ, central bank, politics, climate and weather, sports, crypto-price, misc). For each bucket, record the number of markets, the typical contract count, the bid-ask spread, the fee schedule, and the time to resolution. Output a survey CSV plus a short notes file.
2. **Pick one or two candidate buckets.** You want spreads small relative to a plausible edge (target effective cost under about 3 cents round trip), enough markets per month for a sample (at least about 50 per quarter), and a clean machine-readable resolution source. Likely best: econ data prints (CPI, jobs, GDP, resolvable from BLS or BEA, plenty of markets, decent liquidity) and central-bank decisions. Sports is liquid, but you would be competing with a mature modeling industry. Politics is illiquid and slow to resolve.
3. **Zero-training calibration probe.** For about 100 to 200 already-resolved markets in the chosen bucket, reconstruct the question plus an evidence pack as it would have looked about 24 hours before resolution. This is the fiddly part: you need point-in-time data with no leakage. Ask two or three frontier models (for example Claude, a GPT-class model, Grok) with the standard forecasting prompt (superforecaster framing, proper-scoring explanation, base-rate-first reasoning, overconfidence warning) for a probability. Score it with the calibration harness: Brier, log loss, and a calibration curve, measured against the market mid at that timestamp and against a naive base-rate baseline.
4. **Gate G0.** Proceed only if at least one frontier model, on at least one bucket, beats the market mid on out-of-sample Brier by a margin that clears the round-trip fee. Translate the Brier delta into an expected cents-per-contract and compare directly to the fee. If an off-the-shelf model cannot beat the mid, fine-tuning it will not either, and you are done cheaply.

> Realistic expectation from the literature: frontier LLMs are about as good as the crowd, below superforecasters, and the market mid in a liquid event market is already a sharp crowd. G0 may well fail. That is a fine outcome. It is a two-day answer to a nagging question.

---

## Phase 1: scaffold before model (still no fine-tuning; about 1 week, only if G0 passes)

Goal: capture the cheap wins first. The Mantic result says the research-and-tool scaffold contributed about 3 of its 7 points, and the open repos already have most of the plumbing.

1. **Build the pipeline.** Crib Polyseer for multi-agent evidence gathering plus Bayesian aggregation, and the Kalshi bot repos (`oracle3`, `kalshi-trading-bot-cli`) for the edge-vs-book, fractional-Kelly, and risk-engine layer. Shape: question, then evidence agents, then per-model probability, then ensemble or Bayesian aggregate, then edge versus the live book, then Kelly-fraction size, then a paper order.
2. **Ensemble three or four frontier models** plus a base-rate model. Track per-model and ensemble calibration live on new markets. This is a forward test, no peeking.
3. **Category gating** (the arXiv 2511.18394 lesson): emit a paper trade only where the model has shown calibration in that sub-category. Everything else is a no-trade.
4. **Gate G1.** Require about 6 to 8 weeks of forward paper trading that shows positive simulated PnL after fees and a calibration plot that actually sits on the diagonal, on a real sample of at least about 40 trades. If the scaffold alone clears this, you may not need a fine-tune; ship the scaffold and stop. If it is close but not over, go to Phase 2.

---

## Phase 2: the fine-tune (only if G1 is close but not over; 1 to 2 weeks plus a few hundred dollars)

Goal: reproduce the Mantic recipe at small scale to get a decorrelated, ensemble-valuable forecaster. The value is diversity in the ensemble, not necessarily a standalone winner.

1. **Base model:** an open model that reasons well and is cheap to RL. `gpt-oss-120b` if you can run it (what Mantic used), or a smaller Qwen or Llama reasoning model for a first pass. LoRA, not a full fine-tune.
2. **Training data:** a few thousand resolved binary questions with point-in-time evidence packs, resolution dates after the base model's knowledge cutoff to avoid memorization. Sources: resolved event markets, Metaculus, and a synthetic-question generator (Mantic generated about 10k with an LLM). Reuse the Phase-0 evidence-reconstruction code.
3. **RL setup, copying Mantic:** policy gradient with GRPO-style advantage normalization; reward is Brier score (more stable than log loss, bounded, lower-variance gradients); the model emits a small mixture or point probability through a tool call. Use a managed trainer or a local 1-to-2-GPU setup. Optionally borrow Trading-R1's reverse reasoning distillation to bootstrap good SFT traces before RL, and a volatility or uncertainty-aware reward so the model is not paid for easy questions.
4. **Evaluate** on a held-out, post-cutoff question set: Brier versus the base model, Brier versus frontier models, and the key metric, error correlation with the frontier ensemble. Decorrelation is the win. Then re-run the Phase-1 forward paper test with the fine-tuned model added to the ensemble.
5. **Gate G2.** The fine-tuned-augmented ensemble beats the Phase-1 ensemble on forward paper Brier and on simulated PnL after fees, over a fresh window of at least about 6 weeks. Only then consider real money, and then with phased sizing (1 contract, then 25, then 50, then 100, with a go/no-go check at each step).

---

## Things that will bite you (write these on the wall)

- **Leakage.** Reconstructing "what was knowable at time T" is the hardest part and the easiest place to fool yourself. Treat any too-good Phase-0 result as a leakage bug until proven otherwise.
- **Liquidity and fees.** A 1-to-2-cent Brier-implied edge is real on paper, but event-market spreads plus fees can be larger than that. The edge has to clear the all-in cost, sized at the depth actually available.
- **Sample size.** Event markets resolve slowly. Forty trades is a thin sample; do not over-update on it. This is a months-long forward test, not a weekend backtest, and backtests here are more dangerous than in crypto because there is no continuous price series and resolutions are sparse.
- **Venue constraints.** Much of the open tooling, the liquidity, and the published edges are Polymarket-side. If you are restricted to another venue such as Kalshi, it may be thinner in exactly the political and news categories where LLMs do best. Factor that into the G0 bucket choice.
- **You may already have the answer.** A careful probe that correctly concludes "no edge" is a success, not a failure. A clean negative from Phase 0 is a good outcome and a cheap one.

---

## What this design explicitly does not do

- No fine-tuning for short-horizon crypto. There is no text, it is microstructure-bound, and a language model has no information edge there.
- No building yet another "13 AI agents trade Kalshi 24/7" scaffold for its own sake. Only the evidence-to-probability-to-edge-to-size pipeline, and only as Phase-1 scaffolding for the calibration test.
- No chasing the "LLM-guided RL trades equities" branch. It only ever beats vanilla RL, a weak bar, and it is not the target market.
- No real money before G2.
