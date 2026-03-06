"""
Mag7 Short-Term Trading Dashboard
----------------------------------
Cloud-safe Streamlit app that reads mag7_closing_prices.csv
using a path relative to this file — works locally and on
Streamlit Cloud, Render, Railway, etc.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mag7 Trading Desk",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cloud-safe data loading ───────────────────────────────────────────────────
# __file__ resolves to the script's location on ANY host.
# The CSV must sit in the same directory as app.py when deployed.
DATA_PATH = Path(__file__).parent / "mag7_closing_prices.csv"

TICKERS = ["NVDA", "GOOGL", "META", "AAPL", "AMZN", "MSFT", "TSLA"]

COLORS = {
    "NVDA": "#76b900",
    "GOOGL": "#4285f4",
    "META": "#0866ff",
    "AAPL": "#a2aaad",
    "AMZN": "#ff9900",
    "MSFT": "#00a4ef",
    "TSLA": "#e31937",
}


@st.cache_data(show_spinner="Loading market data…")
def load_data() -> pd.DataFrame:
    """
    Load and validate the CSV.  Raises a clear error if the file is
    missing so the user knows exactly what to fix when deploying.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find '{DATA_PATH.name}'. "
            "Make sure mag7_closing_prices.csv is in the same folder as app.py."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Validate expected columns
    missing = [t for t in TICKERS if t not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected ticker columns: {missing}")

    return df


@st.cache_data(show_spinner=False)
def enrich(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add MA20, MA50, daily return, and RSI(14) for a given ticker."""
    s = df[["Date", ticker]].copy()
    s["MA20"] = s[ticker].rolling(20).mean()
    s["MA50"] = s[ticker].rolling(50).mean()
    s["Return"]  = s[ticker].pct_change() * 100

    # RSI-14
    delta = s[ticker].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    s["RSI"] = 100 - (100 / (1 + rs))

    return s


def signal_badge(price: float, ma20: float) -> str:
    if price > ma20:
        return "🟢 ABOVE MA20"
    return "🔴 BELOW MA20"


# ── Load data (crash early with a helpful message) ────────────────────────────
try:
    df = load_data()
except (FileNotFoundError, ValueError) as e:
    st.error(f"**Data error:** {e}")
    st.stop()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")

    ticker = st.selectbox("Ticker", TICKERS, index=0)

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    show_ma20 = st.checkbox("Show MA20", value=True)
    show_ma50 = st.checkbox("Show MA50", value=True)
    show_rsi  = st.checkbox("Show RSI(14)", value=True)

    st.divider()
    st.caption(
        f"Data: {min_date} → {max_date}  \n"
        f"{len(df):,} trading days  \n"
        f"Source: `{DATA_PATH.name}`"
    )

# ── Filter by date range ──────────────────────────────────────────────────────
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start, end = df["Date"].min(), df["Date"].max()

enriched = enrich(df, ticker)
mask = (enriched["Date"] >= start) & (enriched["Date"] <= end)
view = enriched[mask].copy()

# ── KPI row ───────────────────────────────────────────────────────────────────
latest   = view.iloc[-1]
earliest = view.iloc[0]
period_return = ((latest[ticker] / earliest[ticker]) - 1) * 100

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Last Close",    f"${latest[ticker]:,.2f}")
col2.metric("Period Return", f"{period_return:+.1f}%")
col3.metric("MA20",          f"${latest['MA20']:,.2f}" if not np.isnan(latest["MA20"]) else "N/A")
col4.metric("RSI(14)",       f"{latest['RSI']:.1f}"   if not np.isnan(latest["RSI"])  else "N/A")
col5.metric("Signal",        signal_badge(latest[ticker], latest["MA20"]) if not np.isnan(latest["MA20"]) else "—")

st.divider()

# ── Price + indicator chart ───────────────────────────────────────────────────
rows = 2 if show_rsi else 1
row_heights = [0.7, 0.3] if show_rsi else [1.0]

fig = make_subplots(
    rows=rows, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=row_heights,
)

color = COLORS.get(ticker, "#ffffff")

# Price line
fig.add_trace(
    go.Scatter(
        x=view["Date"], y=view[ticker],
        name=ticker, line=dict(color=color, width=2),
        hovertemplate="%{x|%b %d %Y}<br><b>$%{y:,.2f}</b><extra></extra>",
    ),
    row=1, col=1,
)

if show_ma20:
    fig.add_trace(
        go.Scatter(
            x=view["Date"], y=view["MA20"],
            name="MA20", line=dict(color="#f0c040", width=1.5, dash="dot"),
            hovertemplate="MA20 $%{y:,.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

if show_ma50:
    fig.add_trace(
        go.Scatter(
            x=view["Date"], y=view["MA50"],
            name="MA50", line=dict(color="#c084fc", width=1.5, dash="dash"),
            hovertemplate="MA50 $%{y:,.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

if show_rsi:
    fig.add_trace(
        go.Scatter(
            x=view["Date"], y=view["RSI"],
            name="RSI(14)", line=dict(color="#38bdf8", width=1.5),
            hovertemplate="RSI %{y:.1f}<extra></extra>",
        ),
        row=2, col=1,
    )
    # Overbought / oversold bands
    for level, clr in [(70, "rgba(239,68,68,0.15)"), (30, "rgba(34,197,94,0.15)")]:
        fig.add_hline(y=level, line=dict(color=clr.replace("0.15", "0.6"), dash="dot", width=1), row=2, col=1)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    font=dict(family="monospace", color="#94a3b8"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    margin=dict(l=10, r=10, t=10, b=10),
    height=520 if show_rsi else 380,
)
fig.update_xaxes(showgrid=False, zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor="#1e293b", zeroline=False)

st.plotly_chart(fig, use_container_width=True)

# ── Comparative returns heatmap ───────────────────────────────────────────────
st.subheader("📊 Mag7 — Comparative Returns")

returns = {}
for t in TICKERS:
    s = df.loc[mask, t]
    if len(s) >= 2:
        returns[t] = round(((s.iloc[-1] / s.iloc[0]) - 1) * 100, 2)

ret_df = (
    pd.DataFrame.from_dict(returns, orient="index", columns=["Return (%)"])
    .sort_values("Return (%)", ascending=False)
    .reset_index()
    .rename(columns={"index": "Ticker"})
)

bar_fig = go.Figure(
    go.Bar(
        x=ret_df["Ticker"],
        y=ret_df["Return (%)"],
        marker_color=[
            COLORS.get(t, "#64748b") for t in ret_df["Ticker"]
        ],
        text=[f"{v:+.1f}%" for v in ret_df["Return (%)"]],
        textposition="outside",
        hovertemplate="%{x}: %{y:+.2f}%<extra></extra>",
    )
)
bar_fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    font=dict(family="monospace", color="#94a3b8"),
    margin=dict(l=10, r=10, t=10, b=10),
    height=280,
    showlegend=False,
    yaxis_title="Return (%)",
)
bar_fig.add_hline(y=0, line=dict(color="#475569", width=1))

st.plotly_chart(bar_fig, use_container_width=True)

# ── Raw data table ────────────────────────────────────────────────────────────
with st.expander("🗂  Raw data"):
    st.dataframe(
        view[["Date", ticker, "MA20", "MA50", "Return", "RSI"]]
        .tail(60)
        .set_index("Date")
        .style.format({
            ticker: "${:,.2f}", "MA20": "${:,.2f}", "MA50": "${:,.2f}",
            "Return": "{:+.2f}%", "RSI": "{:.1f}",
        }),
        use_container_width=True,
    )
