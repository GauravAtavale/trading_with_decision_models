# Trading with decision models

Can a model that **doesn't generate text** make trading decisions? This arena gives each
decision model $100 of paper money and asks it, every 15 minutes, whether to buy or sell
BTC and ETH. Every model sees exactly the same market snapshot, and the results are tracked
against simple baselines.

v1 compares **Jev** (TypeSafe) and **Microsoft-Decision-1** on a historical replay of the
last week. It follows the same design as
[decision-models-chess](https://github.com/GauravAtavale/decision-models-chess): the code
does the arithmetic and states the facts in words; the model only judges.

## How a decision works

```
15-min candles ─► features in words ─► same snapshot to every model ─► P(yes) ─► BUY / SELL / HOLD
                 (no prices, no dates)   + that model's own position     two yes/no questions
```

Each request asks two `noul` (yes/no) questions:

| Question | When | Used for |
|---|---|---|
| `up_next`: will the price be >0.3% higher in 1 hour? | every bar, same for every model | forecast quality (Brier, calibration) |
| `act`: should you **buy** now (when in cash) / **sell** now (when holding)? | every bar | the trade |

The model acts when P(yes) ≥ 0.6, the same threshold for every model. Yes/no questions are used
instead of a BUY/HOLD/SELL multiple choice because both vendors document that choice answers can
shift with option order.

**Execution:** a decision made at a bar's close fills at the next bar's open, long-only,
all-in / all-out, with a fee and slippage charged on every side. The $100 is split
equally across the assets.

**The snapshot** (about 500 tokens) contains returns over 15 min / 1 h / 4 h / 24 h, position
versus the 20- and 80-bar averages and their direction, RSI, volatility versus its 7-day norm,
relative volume, where the price sits in its 24 h range, the other coin's moves, hour and
weekday, the model's own position and P&L, and the trading cost. Absolute prices and dates
are left out, so the models can't "remember" what happened on a given day.

## Setup

```bash
git clone https://github.com/GauravAtavale/trading_with_decision_models.git
cd trading_with_decision_models
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in your keys
```

**Keys** (in `.env` or the environment):

| Model | Option A | Option B |
|---|---|---|
| Jev | `TYPESAFE_API_KEY` | `OPENROUTER_API_KEY` |
| Microsoft-Decision-1 | `AZURE_FOUNDRY_ENDPOINT` + `AZURE_API_KEY` + `MS_DECISION_DEPLOYMENT` (deploy it from the Foundry model catalog first) | `OPENROUTER_API_KEY` |

The first provider with a complete set of keys is used; the startup log shows which.

## Run the one-week replay

```bash
python -m tdm probe --model jev           # one real request; check the raw response
python -m tdm probe --model microsoft
python -m tdm replay --dry-run            # sample request + token/cost estimate, no calls
python -m tdm replay --models jev,microsoft
streamlit run dashboard.py
```

The replay downloads about 15 days of 15-minute candles from Coinbase's public API (8 days of
warmup plus the 7-day window), caches them in `data/`, and makes about 1,350 requests per model.
At $0.042 per million input tokens that is a few cents per model. Every API response is cached in
`runs/api_cache.sqlite`, so re-running the same window is free and gives identical results.

Other commands:

```bash
python -m tdm replay --days 3 --end 2026-10-10T00:00   # a different window
python -m tdm report --run latest --compare jev,microsoft
python -m tdm consistency --model jev --n 50           # re-ask 50 bars with JSON keys shuffled
python -m tdm runs                                     # list saved runs
python -m tdm replay --source synthetic --models mock  # offline demo, no keys or network
```

## What gets measured

The report (`reports/<run>.md`) and the dashboard have four layers. Over one week, P&L is
mostly luck; the second layer is the more sensitive measure, as the per-move evaluation
was in the chess project.

1. **Trading:** final value, return versus buy-and-hold, annualized Sharpe ratio, max drawdown,
   trades, win rate, fees paid, time in the market.
2. **Decision quality:**
   - **Brier score and log-loss** on `up_next` (lower is better). Compare against
     `brier_base_rate`, which always predicts the window's up-rate; a model has skill only
     if it beats that.
   - **Calibration chart and ECE:** when a model says 70%, does the price rise 70% of the time?
   - **Hit rate when confident:** accuracy when P is at least 0.65 or at most 0.35.
   - **Regret:** return given up versus the better of the two actions available at each bar.
3. **Reliability:** latency (p50/p95), error rate, tokens and cost, how often the models agree,
   and the shuffled-key flip rate.
4. **Is the gap real?** A paired circular block bootstrap on identical bars, for per-bar returns
   and for Brier scores, with a 95% confidence interval.

Baselines run automatically with the same broker and costs:
- `buy_hold`
- `random`: uniform probabilities with the same 0.6 threshold
- `momentum`

## Layout

```
config.toml          assets, bar size, costs, thresholds, model endpoints
tdm/data.py          Coinbase candles (cached CSV), CSV import, synthetic data
tdm/features.py      features and their wording (no lookahead, no prices or dates)
tdm/snapshot.py      state = market + own position; the two yes/no questions
tdm/clients.py       Jev / Microsoft client (one TypeSafe-style client), mock, baselines
tdm/broker.py        paper broker: next-bar-open fills, fees, slippage
tdm/engine.py        replay loop
tdm/db.py            SQLite schema (runs, snapshots, decisions, fills, equity, outcomes)
tdm/metrics.py       all scores and the paired bootstrap
tdm/report.py        markdown report and the consistency check
dashboard.py         Streamlit dashboard
tests/               offline tests, including a fake API server for both auth styles
```

## Notes and caveats

- **Costs:** `fee_bps = 10` per side is optimistic for retail spot trading. Set it to the
  real fee of the venue you have in mind; frequent trading can easily lose more to fees than
  it gains.
- **Response format:** Jev returns `answers[<name>]["noul"]` as P(yes). The parser also accepts
  `probability` and `probabilities.true`, in case Microsoft's response differs. Run
  `probe` once per model to confirm.
- **Criteria:** if an endpoint rejects the `criteria` field (HTTP 422), the client drops it
  and retries automatically.
- **Contamination:** dates and prices are left out of the snapshot, but a hosted model may
  still know the broad market direction of a past week. A live forward test is the clean test.
- **Synthetic data** tests the plumbing only; it says nothing about real markets.

```bash
python -m pytest -q   # offline: no keys, no network
```
