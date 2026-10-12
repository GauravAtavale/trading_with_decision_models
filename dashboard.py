"""Arena dashboard.   Run:  streamlit run dashboard.py"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tdm import db as dbm
from tdm import metrics as M
from tdm.config import ROOT, load_config

st.set_page_config(page_title="Decision-model trading arena", layout="wide")

# Fixed color per entity (never by rank). Baselines are neutral grays with dash styles.
COLORS = {"jev": "#2a78d6", "microsoft": "#eb6834", "mock": "#1baf7a"}
EXTRA = ["#1baf7a", "#6250d6", "#e87ba4"]
BASELINE_STYLE = {"buy_hold": ("#8a8984", "dash"), "momentum": ("#8a8984", "dot"),
                  "random": ("#b5b4ae", "dashdot")}


def style_for(model: str, i: int = 0):
    if model in BASELINE_STYLE:
        c, dash = BASELINE_STYLE[model]
        return dict(color=c, width=1.5, dash=dash)
    return dict(color=COLORS.get(model, EXTRA[i % len(EXTRA)]), width=2)


def base_layout(fig, title, ytitle, height=360):
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=height,
                      margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
                      yaxis_title=ytitle, template="plotly_white")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


@st.cache_resource
def get_conn(path: str):
    return dbm.connect(Path(path))


cfg = load_config()
db_path = st.sidebar.text_input("Database", str(cfg.db_path))
if not Path(db_path).is_file():
    st.info(f"No database at {db_path} yet. Run `python -m tdm replay` first.")
    st.stop()
conn = get_conn(db_path)
runs = dbm.frame(conn, "SELECT run_id, source, start_ts, end_ts, models, notes FROM runs ORDER BY created_utc DESC")
if runs.empty:
    st.info("No runs yet. Run `python -m tdm replay` first.")
    st.stop()
run_id = st.sidebar.selectbox("Run", runs.run_id, format_func=lambda r: f"{r}")


@st.cache_data(show_spinner=False)
def load(run_id: str, db_path: str):
    return M.load_run(get_conn(db_path), run_id)


d = load(run_id, db_path)
rcfg = d["cfg"]
assets = rcfg.get("assets", [])
api_models = [m for m in d["models"] if m not in BASELINE_STYLE]
asset_pick = st.sidebar.radio("Assets", ["All (portfolio)"] + assets)
shown = st.sidebar.multiselect("Models", d["models"], default=d["models"])

st.title("Decision-model trading arena")
st.caption(f"Run **{run_id}** · {d['run']['start_ts'][:16]} → {d['run']['end_ts'][:16]} UTC · "
           f"source {d['run']['source']} · {rcfg.get('bar_minutes')}-min bars · "
           f"{rcfg.get('fee_bps')} bps fee + {rcfg.get('slippage_bps')} bps slippage per side · "
           f"threshold {rcfg.get('action_threshold')} · ${rcfg.get('seed_usd')} per model")
if d["run"]["source"] == "synthetic":
    st.warning("Synthetic data: this run tests the plumbing and says nothing about real markets.")

# ------------------------------------------------------------------ KPIs
trade = M.trading_table(d)
qual = M.quality_table(d)
rel = M.reliability_table(d)
cols = st.columns(max(1, len(api_models) + 1))
for col, m in zip(cols, api_models + ["buy_hold"]):
    if m not in trade.index:
        continue
    t = trade.loc[m]
    delta = None if m == "buy_hold" else f"{t['vs_buy_hold_pp']:+.2f} pp vs buy-and-hold"
    col.metric(m, f"${t['final_$']:,.2f}", delta)
    if m in qual.index and "brier" in qual.columns and np.isfinite(qual.loc[m].get("brier", np.nan)):
        col.caption(f"Brier {qual.loc[m, 'brier']:.3f} (base rate {qual.loc[m, 'brier_base_rate']:.3f}) · "
                    f"{int(trade.loc[m, 'trades'])} trades")

# ------------------------------------------------------------------ equity + drawdown
eq = d["equity"]
if asset_pick != "All (portfolio)":
    eq = eq[eq.asset == asset_pick]
wide = M.total_equity(eq)
wide.index = pd.to_datetime(wide.index)
fig = go.Figure()
for i, m in enumerate(shown):
    if m in wide:
        fig.add_trace(go.Scatter(x=wide.index, y=wide[m], name=m, mode="lines", line=style_for(m, i),
                                 hovertemplate="%{y:$,.2f}"))
st.plotly_chart(base_layout(fig, f"Equity — {asset_pick}", "USD"), width="stretch")

dd = wide / wide.cummax() - 1
fig = go.Figure()
for i, m in enumerate(shown):
    if m in dd:
        fig.add_trace(go.Scatter(x=dd.index, y=dd[m] * 100, name=m, mode="lines", line=style_for(m, i),
                                 hovertemplate="%{y:.2f}%"))
st.plotly_chart(base_layout(fig, "Drawdown from peak", "%", 260), width="stretch")

# ------------------------------------------------------------------ tables
st.subheader("Scorecards")
tab1, tab2, tab3, tab4 = st.tabs(["Trading", "Decision quality", "Reliability", "Is the gap real?"])
with tab1:
    st.dataframe(trade.loc[[m for m in shown if m in trade.index]].style.format(precision=2),
                 width="stretch")
    st.caption("Sharpe is annualized from per-bar returns, so it swings wildly over a one-week window.")
with tab2:
    st.dataframe(qual.loc[[m for m in shown if m in qual.index]].style.format(precision=3),
                 width="stretch")
    st.caption("Brier/log-loss score the `up_next` forecast (lower is better). Beat `brier_base_rate` "
               "to show any skill. Regret = return given up vs the better action at each bar.")
with tab3:
    st.dataframe(rel.loc[[m for m in shown if m in rel.index]].style.format(precision=1),
                 width="stretch")
    st.markdown("**Agreement on `up_next`** (% of bars where two models lean the same way)")
    st.dataframe(M.agreement_table(d).style.format("{:.0f}"), width="stretch")
with tab4:
    pairs = []
    if len(api_models) >= 2:
        pairs.append((api_models[0], api_models[1]))
    pairs += [(m, "buy_hold") for m in api_models]
    rows = []
    for a, b in pairs:
        c = M.paired_comparison(d, a, b)
        rows.append({"comparison": f"{a} vs {b}", "bars": c["bars"],
                     "return gap (pp)": c["return_diff_pp"],
                     "per-bar gap (bps)": c["per_bar_diff_bps"],
                     "95% CI low": c["ci95_bps"][0], "95% CI high": c["ci95_bps"][1],
                     f"P(first not better)": c["p_a_not_better"],
                     "Brier gain": c.get("brier_gain", np.nan)})
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("comparison").style.format(precision=3),
                     width="stretch")
        st.caption("Paired circular block bootstrap on identical bars. A CI that spans 0 means the "
                   "window can't separate the two.")

# ------------------------------------------------------------------ calibration
st.subheader("Calibration of the `up_next` forecast")
j = d["decisions"].merge(d["outcomes"][["asset", "ts", "up_label"]], on=["asset", "ts"]).dropna(
    subset=["p_up", "up_label"])
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                         line=dict(color="#b5b4ae", width=1, dash="dash"), hoverinfo="skip"))
for i, m in enumerate(shown):
    x = j[j.model == m]
    if x.empty:
        continue
    cal = M.calibration_bins(x.p_up, x.up_label.astype(float))
    cal = cal[cal.n >= 5]
    fig.add_trace(go.Scatter(x=cal.mean_p, y=cal.observed, name=m, mode="lines+markers",
                             line=style_for(m, i), marker=dict(size=9),
                             customdata=np.stack([cal.n, cal["range"]], axis=1),
                             hovertemplate="said %{x:.2f} · happened %{y:.2f} · n=%{customdata[0]}"))
fig = base_layout(fig, "When a model says p, how often does the price actually rise?", "observed frequency", 380)
fig.update_layout(hovermode="closest")
fig.update_xaxes(title="predicted probability", range=[0, 1])
fig.update_yaxes(range=[0, 1])
st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ trades on price
st.subheader("Trades on the price chart")
c1, c2 = st.columns(2)
pick_asset = c1.selectbox("Asset", assets, key="trade_asset")
pick_model = c2.selectbox("Model", d["models"], key="trade_model")
bars = d["bars"][d["bars"].asset == pick_asset].copy()
bars["ts"] = pd.to_datetime(bars.ts)
fills = d["fills"][(d["fills"].asset == pick_asset) & (d["fills"].model == pick_model)].copy()
fills["ts"] = pd.to_datetime(fills.ts)
fig = go.Figure()
fig.add_trace(go.Scatter(x=bars.ts, y=bars.close, name=pick_asset, mode="lines",
                         line=dict(color="#52514e", width=1.5), hovertemplate="%{y:,.2f}"))
for side, symbol, color in (("BUY", "triangle-up", "#1baf7a"), ("SELL", "triangle-down", "#e34948")):
    f = fills[fills.side == side]
    fig.add_trace(go.Scatter(x=f.ts, y=f.price, name=side.lower(), mode="markers",
                             marker=dict(symbol=symbol, size=11, color=color,
                                         line=dict(color="white", width=1.5)),
                             hovertemplate=side + " at %{y:,.2f}"))
st.plotly_chart(base_layout(fig, f"{pick_model} on {pick_asset}", "price", 340), width="stretch")

# ------------------------------------------------------------------ decision explorer
st.subheader("Decision log")
dec = d["decisions"].merge(d["outcomes"][["asset", "ts", "fwd_ret", "up_label"]], on=["asset", "ts"], how="left")
c1, c2, c3 = st.columns(3)
m_pick = c1.selectbox("Model", d["models"], key="log_model")
a_pick = c2.selectbox("Asset", assets, key="log_asset")
only = c3.selectbox("Show", ["all bars", "trades only", "errors only"])
view = dec[(dec.model == m_pick) & (dec.asset == a_pick)]
if only == "trades only":
    view = view[view.action != "HOLD"]
elif only == "errors only":
    view = view[view.error.notna()]
view = view[["ts", "position", "p_up", "p_act", "action", "fwd_ret", "up_label", "latency_ms", "cached", "error"]]
view = view.assign(fwd_ret=view.fwd_ret * 100).rename(columns={"fwd_ret": "next_hour_%"})
st.dataframe(view.set_index("ts").style.format(precision=3, na_rep="–"), width="stretch", height=320)
if len(view):
    ts_pick = st.selectbox("Inspect the exact request for bar", view.ts.tolist()[::-1])
    row = conn.execute("SELECT state_json, questions_json FROM decisions WHERE run_id=? AND model=? "
                       "AND asset=? AND ts=?", (run_id, m_pick, a_pick, ts_pick)).fetchone()
    if row and row[0]:
        c1, c2 = st.columns(2)
        c1.markdown("**state**")
        c1.json(json.loads(row[0]))
        c2.markdown("**questions**")
        c2.json(json.loads(row[1]))
    else:
        st.caption("Baselines don't send requests, so no state is stored for them.")
