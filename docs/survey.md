# Survey: fine-tuning LLMs to trade stocks and prediction markets

Survey current as of mid-2026. Sources are linked inline, with a consolidated list at the end. Every URL was checked; see the note above the source list for the link-check result.

## 0. How to read the field

"Retrained an LLM to trade" almost always means one of four things, in increasing order of how much the training actually matters.

1. **Prompt-engineered agent, no training.** An off-the-shelf model (GPT-4-class, Grok-4, Claude) wrapped in a scaffold: news scraper, then a "you are a superforecaster" prompt, then a probability, then Kelly sizing, then an order. Most "AI trading bot" repos are this. Nothing is trained. The edge, if any, lives in the scaffold and the data plumbing, not the model.
2. **Light domain fine-tune for understanding, not decisions.** FinGPT is the example: LoRA-tune an open model on financial-sentiment data so it labels headlines well, then let a separate rule turn sentiment into trades. The fine-tune improves an NLP sub-task. The trading layer is unchanged.
3. **SFT plus RL to produce trading reasoning and decisions.** Trading-R1 is the example: distill reasoning traces, then RL with a return-and-volatility-aware reward so the model emits a structured thesis plus a discretized position. Here the training output is the trade signal.
4. **RL fine-tune to produce calibrated probabilities on resolvable questions.** Mantic's `gpt-oss-120b` forecasting model and the Metaculus AI-benchmark bots are the examples. The reward is a proper scoring rule (Brier or log) against realized outcomes. This is the only branch where "we trained an LLM and it measurably got better at the thing prediction markets price" is cleanly true.

Branches 1 through 3 are the "stock trading" headlines. Branch 4 is the "prediction market" headline. They have little to do with each other technically, and branch 4 is the one that holds up.

---

## 1. Stock and crypto trading LLMs

### 1.1 Trading-R1 (Tauric Research, 2025), the most serious RL-trained trading reasoner

Method. arXiv 2509.11420. SFT, then a three-stage easy-to-hard RL curriculum. The dataset, Tauric-TR1-DB, is about 100k samples over 18 months (January 2024 to May 2025), 14 tickers, five heterogeneous data sources (prices, news, filings, and others). It uses reverse reasoning distillation, working backward from a known outcome to a plausible thesis, to bootstrap the SFT data. It uses a volatility-aware reward so the model is not paid for luck in a calm regime. Output is a structured evidence-based thesis plus a discretized strong-sell to strong-buy action. The paper reports better risk-adjusted returns and lower drawdown than instruction and reasoning baselines on six equities and ETFs.

Grade: real research, oversold as alpha. This is genuinely the state of the art for this branch, and two tricks are worth stealing: the volatility-aware reward and the reverse distillation. The limits are real. It is equities at a daily-ish horizon, the evaluation window is small and overlaps the training regime, and "beats other LLMs" is not "beats a cheap quant baseline or buy-and-hold after costs." There are no live-money results. The honest selling point is interpretability, a thesis you can audit, not edge.

### 1.2 FinGPT / FinMem / FinRobot (AI4Finance Foundation), the popular open-source stack

- **FinGPT** (arXiv 2306.06031): LoRA-tune open models on financial-sentiment data from news and tweets. Cheap, under $300 per run on a single RTX 3090. It is a sentiment labeler, full stop. The "FinGPT trader" repos bolt a sentiment-to-signal rule on top.
- **FinMem** (NeurIPS workshop; repo `pipiku915/FinMem-LLM-StockTrading`): an agent with layered memory (short, mid, long) and a character/risk-profile module. The base model is not fine-tuned; the innovation is the memory architecture feeding the prompt. It reports strong single-name backtests.
- **FinRobot**: a platform and agent-orchestration layer, with tutorials for price prediction and analysis.

Grade: a parts bin, not evidence. Useful as plumbing and as a sentiment component. None of it shows that an LLM has a trading edge. The backtests are single-name, short, and cost-light, and the authors are upfront that this is a research platform.

### 1.3 LLM-guided RL (LLM proposes strategy, RL agent executes)

The common shape: the LLM emits a high-level strategy or a per-step instruction conditioned on news and market state, a classical RL agent (PPO or DQN) trades, and the LLM context improves return and risk versus unguided RL. Representative papers: "Language Model Guided Reinforcement Learning in Quantitative Trading" (arXiv 2508.02366); "Integrating LLMs and RL for Sentiment-Driven Quantitative Trading" (arXiv 2510.10526); "Advancing Algorithmic Trading with LLMs," also circulated as Stock-Evol-Instruct (OpenReview).

Grade: honest, but a weak bar. The ablations are candid that the LLM contributes context, not magic. The comparison is to vanilla RL. "Better than a context-blind PPO agent" is not "has alpha."

### 1.4 The "I built N AI agents that trade Kalshi or Polymarket 24/7" genre (blogs and repos)

The pattern is a multi-agent ensemble (forecaster, news, bull, bear, risk-manager), each an off-the-shelf LLM, that debates, aggregates to a probability, sizes with fractional Kelly, and sends a paper or live order under hard daily loss caps. Examples: `ryanfrigo/kalshi-ai-trading-bot`, `OctagonAI/kalshi-trading-bot-cli`, `YichengYang-Ethan/oracle3`, the Dev.to "13 AI agents" post, and the Dev Genius "two-layer AI system" post. `oracle3` is the most quant-flavored: a Wang-transform pricing engine calibrated on 291k contracts, with the LLM as a side input rather than the core.

Grade: zero training, scaffolds only. These are worth cribbing for the risk engine and the Kelly and loss-cap plumbing. The performance claims are self-reported, short, and unaudited. The often-repeated "14 of the top-20 Polymarket wallets are bots" is true, but those bots are overwhelmingly arbitrage and latency bots, not LLM forecasters. That is a latency edge, not an LLM edge.

### 1.5 Live model-vs-model trading arenas

nof1 and "Alpha Arena"-style events put frontier models head-to-head trading real but small capital on crypto. Entertaining, tiny samples, and mostly a measure of who blows up slowest. Not evidence of a trainable edge, though worth watching for which prompts and tools the better entrants use.

**Bottom line on branches 1 through 3.** The literature is real research and the open-source stacks are real code. None of it establishes that an LLM, trained or not, beats a competent cheap baseline on price direction after costs. The honest wins are narrow: better text understanding (FinGPT), interpretable theses (Trading-R1), and ensemble diversity.

A short-horizon crypto contract, for example a fifteen-minute BTC price market, is the worst case for this technology. It carries no text and no fundamentals, and it is dominated by microstructure. The information set at that horizon is essentially a price series, and a language model adds nothing to it. That is a latency and market-making problem, not a language-model problem.

---

## 2. Prediction-market and event-forecasting LLMs, the part that works

### 2.1 Mantic and Thinking Machines Lab, "Training LLMs to Predict World Events" (2026)

Method. They RL-fine-tuned `gpt-oss-120b` to output a probability, actually a small mixture model via tool calls, on binary world-event questions, inside a two-phase system: deep-research agents gather evidence, then the tuned model reasons over it and emits the distribution. The optimizer is policy gradient with GRPO-style advantage normalization. The reward is Brier score, chosen over log score for lower-variance, bounded-range gradients and more stable training. Training used the Tinker API, batch 64, group 8. Training data was about 10k LLM-generated binary questions from August 2024 to December 2025, with outcomes known but past the model's knowledge cutoff. The test set was the Q2-2025 Metaculus AI Benchmark, roughly 200 unseen questions.

Results. Base `gpt-oss-120b` scored 38.6 and the fine-tuned model scored 45.8 baseline points, a gain of about 7, which roughly matched Gemini 3 Pro. In the best ensemble the fine-tuned model earns 40 to 56 percent weight, the "least replaceable" contributor alongside Grok 4, because its errors are decorrelated from the frontier models. The authors are explicit about one caveat: the model alone, with no research phase and no tools, gained only about 3 points rather than 7, so the scaffold does a lot of the work.

Grade: the cleanest existence proof in the field. On-task RL took a mediocre open model to near-frontier forecasting accuracy and made it ensemble-valuable. It is directly relevant to any resolvable yes/no market with textual evidence behind it. It is not relevant to short-horizon crypto.

### 2.2 Metaculus AI Benchmark tournaments (2024 to 2025), the ongoing scoreboard

- Quarterly bot-vs-pro tournaments. In Q4 2024 the top bot "team" lost to Metaculus Pro forecasters, but not significantly (p about 0.079, log scoring, weighted t-test). Both were well-calibrated, and the Pros won on discrimination, not calibration. The gap has narrowed each quarter.
- "Evaluating LLMs on Real-World Forecasting Against Expert Forecasters" (arXiv 2507.04562): the best LLM scored Brier 0.135, beating the Metaculus crowd at 0.149 but behind the superforecaster panel at 0.122. LLMs are typically overconfident, and worst on events they think are likely.

Grade: frontier LLMs land at about the level of an engaged human, below superforecasters. Fine-tuning (see 2.1) closes the gap to frontier with a small open model. It does not yet beat superforecasters. Standard prompt hygiene helps: a superforecaster framing, an explanation of proper scoring, a list of forecasting principles, a warning about historical overconfidence, and forced step-by-step base-rate-first reasoning.

### 2.3 "Forecasting ability depends on what you are asking" (arXiv 2511.18394, OpenReview)

Claim. LLM forecasting skill is highly category-dependent: decent on some domains, near-random on others. Do not evaluate or deploy on an undifferentiated question pool. Segment by category, and only trade where the model is actually calibrated.

Grade: the single most actionable finding for a real deployment. Pick the narrow slice of markets where the model demonstrably calibrates, and ignore the rest.

### 2.4 Applied tools (mostly scaffolds, useful as reference architectures)

- "I Built an LLM Tool That Beats Human Intuition on Polymarket" (Bokarev, Medium): an Alphascope-style pipeline, an LLM plus curated sources plus forced step-by-step, producing a structured prediction. The edge-vs-human claim is self-reported.
- **Polyseer** (open source): multi-agent evidence gathering plus Bayesian aggregation for Polymarket and Kalshi research. A good aggregation reference. (Named here as in the source material; no canonical URL is cited, so none is linked.)
- **Prediction Arena** (arXiv 2604.07355) and similar mispricing dashboards: benchmarks of models against live market prices. Useful for the edge-vs-order-book computation.
- `aarora4/Awesome-Prediction-Market-Tools`: a curated index of the above.

Grade: a parts catalog for the research-to-probability-to-edge-to-size pipeline. The fine-tuning piece (2.1) plugs in as the probability stage. Most of the rest is already built somewhere you can read.

---

## 3. What this implies for building

| Question | Verdict |
|---|---|
| Fine-tune an LLM to trade short-horizon crypto (for example a 15-minute BTC contract)? | No. No text, microstructure-dominated. An LLM has no information edge at that horizon. This is a latency and market-making game, not a language-model game. |
| Use an LLM as a sentiment or feature source for short-horizon crypto? | Marginal at best. At a 15-minute horizon there is too little text per window for sentiment to add signal. Skip it. |
| Fine-tune or scaffold an LLM for longer-horizon event markets (econ prints, central-bank decisions, politics and legislation, sports, "by date Y")? | Maybe. This is the only branch worth a probe. It is the question class where sections 2.1 through 2.3 show a real, ensemble-valuable edge over the crowd. Constraints: the edge is small and below superforecasters; it is highly category-dependent, so you must find the calibrated slice; event-market liquidity and fees can eat a small edge; and if you are restricted to one venue, the categories where LLMs do best may be the thin ones there. |
| Cheapest way to find out? | A LoRA or small RL fine-tune costs a few hundred dollars (FinGPT scale) up to a managed training run (Mantic scale). The bottleneck is data plus a leakage-free calibration harness, not compute. |

The experiment design ([experiment-design.md](experiment-design.md)) takes the one "maybe" and puts a zero-training decision gate in front of it, before any model is touched.

---

## Sources

Link-check result (last verified for this survey): 37 of 38 URLs live, 0 redirected, 1 dead. The dead link is `muxprotocol/kalshi-trading-bot`, annotated below. Three live links (the Metaculus tournament page, the Bokarev Medium post, and the Dev Genius post) return an anti-bot 403 to automated fetchers but load normally in a browser at the canonical URLs given.

Trading and RL:
- [Trading-R1: Financial Trading with LLM Reasoning via Reinforcement Learning (arXiv 2509.11420)](https://arxiv.org/abs/2509.11420) - [PDF](https://arxiv.org/pdf/2509.11420) - [Moonlight review](https://www.themoonlight.io/en/review/trading-r1-financial-trading-with-llm-reasoning-via-reinforcement-learning)
- [Language Model Guided RL in Quantitative Trading (arXiv 2508.02366)](https://arxiv.org/html/2508.02366v2)
- [Integrating LLMs and RL for Sentiment-Driven Quantitative Trading (arXiv 2510.10526)](https://arxiv.org/html/2510.10526v1)
- [Advancing Algorithmic Trading with LLMs / "Stock-Evol-Instruct" (OpenReview)](https://openreview.net/forum?id=w7BGq6ozOL)
- [LLMs in equity markets: applications, techniques, insights (Frontiers in AI, 2025)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1608365/full) - [PMC mirror](https://pmc.ncbi.nlm.nih.gov/articles/PMC12421730/)
- [Top 3 LLM+RL Advances in Equity Trading (2025), Slava Nesterov](http://www.slavanesterov.com/2025/05/3-llmrl-advances-in-equity-trading-2025.html)
- [FinGPT (GitHub, AI4Finance)](https://github.com/AI4Finance-Foundation/FinGPT) - [arXiv 2306.06031](https://arxiv.org/html/2306.06031v2) - [HF models](https://huggingface.co/FinGPT)
- [FinMem-LLM-StockTrading (GitHub)](https://github.com/pipiku915/FinMem-LLM-StockTrading)
- [FinRobot (GitHub, AI4Finance)](https://github.com/ai4finance-foundation/finrobot)
- [LLM-Enhanced-Trading (GitHub, Ronitt272)](https://github.com/Ronitt272/LLM-Enhanced-Trading) - [fingpt_trader (GitHub, ashioyajotham)](https://github.com/ashioyajotham/fingpt_trader)

Kalshi and Polymarket agents (scaffolds):
- [kalshi-ai-trading-bot (GitHub, ryanfrigo)](https://github.com/ryanfrigo/kalshi-ai-trading-bot)
- kalshi-trading-bot (GitHub, muxprotocol): link dead. `https://github.com/muxprotocol/kalshi-trading-bot` returns HTTP 404 as of the last link check; the repo appears to have been removed and no replacement location was found.
- [kalshi-trading-bot-cli (GitHub, OctagonAI)](https://github.com/OctagonAI/kalshi-trading-bot-cli)
- [oracle3, autonomous prediction-market agent (GitHub, YichengYang-Ethan)](https://github.com/YichengYang-Ethan/oracle3)
- [I built 13 AI agents that trade Kalshi 24/7 (Dev.to)](https://dev.to/bearware/i-built-13-ai-agents-that-trade-kalshi-prediction-markets-247-heres-how-it-works-23k9)
- [Two-layer AI system trading Polymarket and Kalshi (Dev Genius)](https://blog.devgenius.io/just-built-a-two-layer-ai-system-that-trades-polymarket-and-kalshi-while-i-sleep-heres-the-aa59ead275f6)
- [AI Agents in Prediction Markets: How Bots Beat Humans (NYC Servers blog)](https://newyorkcityservers.com/blog/ai-agents-prediction-market-trading)
- [Kalshi Trading Bot with LLM Extraction (Mixpeek)](https://mixpeek.com/blog/kalshi-trading-bot-semantic-search-llm-extraction)

Forecasting LLMs (the part that works):
- [Training LLMs to Predict World Events, Mantic and Thinking Machines Lab](https://thinkingmachines.ai/news/training-llms-to-predict-world-events/)
- [Metaculus AI Forecasting Benchmark Tournament](https://www.metaculus.com/aib/) - [Q4 2024 results (EA Forum)](https://forum.effectivealtruism.org/posts/TG2zCDCozMcDLgoJ5/metaculus-q4-ai-benchmarking-bots-are-closing-the-gap) - [LessWrong mirror](https://www.lesswrong.com/posts/P8YwCvHoF2FHQoHjF/metaculus-q4-ai-benchmarking-bots-are-closing-the-gap) - [Q2 results (EA Forum)](https://forum.effectivealtruism.org/posts/F2stjK9wHSy3HPEC9/q2-ai-benchmark-results-pros-maintain-clear-lead)
- [Evaluating LLMs on Real-World Forecasting Against Expert Forecasters (arXiv 2507.04562)](https://arxiv.org/html/2507.04562v3) - [PDF](https://arxiv.org/pdf/2507.04562)
- [Future Is Unevenly Distributed: Forecasting Ability of LLMs Depends on What We're Asking (arXiv 2511.18394)](https://arxiv.org/html/2511.18394v1) - [OpenReview](https://openreview.net/pdf?id=zzF5H0kZ8I)
- [Prediction Arena: Benchmarking AI Models on Real-World Prediction Markets (arXiv 2604.07355)](https://arxiv.org/html/2604.07355v1)
- [I Built an LLM Tool That Beats Human Intuition on Polymarket (Bokarev, Medium)](https://bokarevs.medium.com/i-built-an-llm-tool-that-beats-human-intuition-on-polymarket-c60f1667b8af)
- [Awesome-Prediction-Market-Tools (GitHub, aarora4)](https://github.com/aarora4/Awesome-Prediction-Market-Tools)
- [Superforecasters: Metrics and Methods (Emergent Mind)](https://www.emergentmind.com/topics/superforecasters)
- [Language Models Still Can't Forecast (Telling the Future, Substack)](https://tellingthefuture.substack.com/p/language-models-still-cant-forecast)
