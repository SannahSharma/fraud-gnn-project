"""
dashboard.py

Purpose: An interactive Streamlit dashboard for exploring the fraud
detection system -- headline metrics, a searchable graph neighborhood
explorer (visualizing WHY an account is flagged, by showing who it's
connected to), and a table of the most suspicious detected fraud rings.

Reads the slim files in dashboard_data/ (created by
scripts/export_dashboard_data.py), so it runs without PyTorch and can be
deployed to Streamlit Community Cloud.

Run with: streamlit run dashboard.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="Fraud Detection & Transaction Intelligence",
    page_icon="🕵️",
    layout="wide",
)

# ------------------------------------------------------------
# Custom design system: fonts, ambient glow background, glass
# cards with hover-lift, gradient text, pulsing status pill.
# NOTE: no blank lines inside this CSS block -- Streamlit's markdown
# parser treats a blank line as the end of a raw HTML block, so any
# blank line here would cause the rest of the CSS to print as plain text.
# ------------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.stApp {
    background:
        radial-gradient(circle at 15% 15%, rgba(0,217,192,0.13) 0%, transparent 42%),
        radial-gradient(circle at 88% 8%, rgba(139,92,246,0.12) 0%, transparent 40%),
        radial-gradient(circle at 50% 100%, rgba(0,217,192,0.06) 0%, transparent 50%),
        #0a0e17;
}
@keyframes fadeInUp { from {opacity:0; transform:translateY(14px);} to {opacity:1; transform:translateY(0);} }
.main .block-container { animation: fadeInUp 0.7s ease both; padding-top: 2rem; }
.status-pill {
    display:inline-flex; align-items:center; gap:6px;
    padding:6px 14px; border-radius:999px;
    background:rgba(0,217,192,0.1); border:1px solid rgba(0,217,192,0.35);
    color:#00D9C0; font-size:0.72rem; letter-spacing:0.08em; text-transform:uppercase;
    font-weight:600; margin-bottom:18px;
}
.pulse-dot {
    width:7px; height:7px; border-radius:50%; background:#00D9C0;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(0,217,192,0.55); }
    70% { box-shadow: 0 0 0 9px rgba(0,217,192,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,217,192,0); }
}
.hero-title {
    font-family:'Space Grotesk', sans-serif; font-weight:700;
    font-size:3rem; line-height:1.08; letter-spacing:-0.02em; margin:0 0 10px 0;
}
.grad-text {
    background: linear-gradient(90deg, #00D9C0, #8B5CF6);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub { color:rgba(250,250,250,0.65); font-size:1.02rem; max-width:640px; margin-bottom:0; }
.kicker { font-size:11px; text-transform:uppercase; letter-spacing:0.14em; color:#00D9C0; font-weight:700; margin-bottom:2px; }
.section-title { font-family:'Space Grotesk', sans-serif; font-size:1.5rem; font-weight:700; margin:0 0 14px 0; }
[data-testid="stMetric"] {
    background: rgba(22,26,37,0.65);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 18px 18px 10px 18px;
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    border-color: rgba(0,217,192,0.45);
    box-shadow: 0 14px 28px rgba(0,217,192,0.10);
}
[data-testid="stMetricValue"] {
    font-family:'Space Grotesk', sans-serif; font-weight:700;
    background: linear-gradient(90deg, #00D9C0, #8B5CF6);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
[data-testid="stMetricLabel"] { font-size:11.5px; text-transform:uppercase; letter-spacing:0.07em; opacity:0.65; }
div[data-baseweb="select"] > div, .stTextInput > div > div > input {
    background: rgba(22,26,37,0.7) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
}
.stAlert { background: rgba(0,217,192,0.08) !important; border: 1px solid rgba(0,217,192,0.28) !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# Extra styles for tabs and insight cards (no blank lines inside, see note above).
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid rgba(255,255,255,0.08); }
.stTabs [data-baseweb="tab"] {
    background: rgba(22,26,37,0.55); border: 1px solid rgba(255,255,255,0.08); border-bottom: none;
    border-radius: 10px 10px 0 0; padding: 10px 18px; font-weight: 600;
}
.stTabs [aria-selected="true"] { background: rgba(0,217,192,0.10); border-color: rgba(0,217,192,0.40); }
.insight {
    background: rgba(22,26,37,0.65); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 18px 20px; height: 100%;
}
.insight .tag { font-size: 11px; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700; color: #A78BFA; }
.insight h4 { font-family: 'Space Grotesk', sans-serif; margin: 6px 0 8px 0; font-size: 1.12rem; }
.insight p { color: rgba(250,250,250,0.72); font-size: 0.93rem; line-height: 1.55; margin: 0; }
.insight .big { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; color: #FAFAFA; }
.chart-note { color: rgba(250,250,250,0.6); font-size: 0.88rem; margin-top: -6px; }
</style>
""", unsafe_allow_html=True)


def section_header(kicker: str, title: str):
    """Render a small uppercase 'kicker' label above a bold section title --
    an editorial-style heading, used instead of plain st.subheader."""
    st.markdown(f'<div class="kicker">{kicker}</div><div class="section-title">{title}</div>',
                unsafe_allow_html=True)


def insight_card(tag: str, title: str, body: str, big: str = ""):
    big_html = f'<div class="big">{big}</div>' if big else ""
    st.markdown(f'<div class="insight"><div class="tag">{tag}</div>{big_html}'
                f'<h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


# Chart colours: validated for colour-blind separation on the dark background.
CLEAN = "#14A89A"     # legitimate / clean
FRAUD = "#E8506F"     # fraud
ACCENT = "#8F6FE8"    # neutral highlight
MUTED = "rgba(250,250,250,0.28)"
INK = "#FAFAFA"
INK_SOFT = "rgba(250,250,250,0.65)"


def style(fig, height=360, legend=True):
    """Shared dark styling so every chart matches the dashboard theme."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=44 if legend else 10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=INK, size=13),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=INK_SOFT)),
        hoverlabel=dict(bgcolor="#161A25", bordercolor="rgba(255,255,255,0.15)",
                        font=dict(family="Inter, sans-serif", color=INK)),
        barcornerradius=4,
        bargap=0.25,
    )
    fig.update_xaxes(automargin=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False, linecolor="rgba(255,255,255,0.12)",
                     tickfont=dict(color=INK_SOFT), title_font=dict(color=INK_SOFT))
    fig.update_yaxes(automargin=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False, linecolor="rgba(255,255,255,0.12)",
                     tickfont=dict(color=INK_SOFT), title_font=dict(color=INK_SOFT))
    return fig


# Hero section
st.markdown("""
<div class="status-pill"><span class="pulse-dot"></span> Live System</div>
<div class="hero-title">Graph-Based Fraud <span class="grad-text">Intelligence</span></div>
<p class="hero-sub">GraphSAGE + XGBoost hybrid scoring, with community detection surfacing
connected fraud rings that a flat, per-transaction model has no way to see.</p>
""", unsafe_allow_html=True)

st.write("")

# ============================================================
# Load everything ONCE per session (Streamlit caches this automatically)
# ============================================================
DASH = Path(__file__).resolve().parent / "dashboard_data"


@st.cache_resource
def load_data():
    nodes = pd.read_parquet(DASH / "nodes.parquet").set_index("node")
    edges = pd.read_parquet(DASH / "edges.parquet")
    rings = pd.read_parquet(DASH / "rings.parquet")
    results = json.loads((DASH / "stats.json").read_text())
    comm_path = DASH / "communities.parquet"
    fi_path = DASH / "feature_importance.parquet"
    comm = pd.read_parquet(comm_path) if comm_path.exists() else None
    fi = pd.read_parquet(fi_path) if fi_path.exists() else None
    return nodes, edges, rings, results, comm, fi


if not (DASH / "nodes.parquet").exists():
    st.error("dashboard_data/ not found. Run `python scripts/export_dashboard_data.py` first.")
    st.stop()

nodes, edges, rings, results, comm, fi = load_data()
score_of = nodes["score"]
label_of = nodes["label"]


def neighborhood(account: str, max_nodes: int = 120):
    """2-hop neighborhood of an account, using the edge list directly
    (no full NetworkX graph in memory). Returns the subgraph, the number of
    direct connections, and whether it was truncated."""
    hop1 = edges[(edges["src"] == account) | (edges["dst"] == account)]
    n1 = set(hop1["src"]) | set(hop1["dst"])
    hop2 = edges[edges["src"].isin(n1) | edges["dst"].isin(n1)]
    members = set(hop2["src"]) | set(hop2["dst"]) | {account}
    truncated = False
    if len(members) > max_nodes:
        others = score_of.reindex(list(members - {account})).fillna(0.0)
        members = {account} | set(others.nlargest(max_nodes - 1).index)
        truncated = True
    sub = edges.loc[edges["src"].isin(members) & edges["dst"].isin(members), ["src", "dst"]]
    ego = nx.Graph()
    ego.add_nodes_from(members)
    ego.add_edges_from(sub.itertuples(index=False, name=None))
    return ego, len(n1 - {account}), truncated


# ============================================================
# KPI row
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accounts (nodes)", f"{results.get('num_nodes', len(nodes)):,}")
col2.metric("Transactions (edges)", f"{results.get('num_edges', len(edges)):,}")
col3.metric("Tabular-only PR-AUC", f"{results.get('hardened_tabular_pr_auc', 0):.3f}")
col4.metric("Hybrid PR-AUC", f"{results.get('score_hybrid_pr_auc', 0):.3f}")
col5.metric("Ring-detection recall", f"{results.get('ring_detection_recall', 0) * 100:.1f}%")

st.write("")

tab_inv, tab_models, tab_patterns, tab_rings, tab_findings = st.tabs(
    ["🔎 Investigate", "📊 Model results", "💸 Fraud patterns", "🕸️ Ring detection", "💡 Key findings"]
)

# ============================================================
# TAB 1: neighborhood explorer (left) + suspicious rings table (right)
# ============================================================
with tab_inv:
    st.write("")
    left, right = st.columns([2, 1])

    with left:
        section_header("Investigate", "Account Neighborhood Explorer")
        st.write("Search any account ID to see its 2-hop transaction neighborhood, "
                 "colored by GraphSAGE fraud risk score.")

        suggested_accounts = score_of.nlargest(25).index.tolist()
        account_id = st.selectbox(
            "Choose a high-risk account, or type any account ID below:",
            options=suggested_accounts,
            index=0,
        )
        custom_id = st.text_input("...or search for a specific account ID:")
        if custom_id.strip():
            account_id = custom_id.strip()

        if account_id not in score_of.index:
            st.error(f"Account '{account_id}' was not found in the graph.")
        else:
            MAX_NODES = 120
            ego, direct_links, truncated = neighborhood(account_id, MAX_NODES)
            if truncated:
                st.caption(f"Neighborhood truncated to the {MAX_NODES} highest-risk accounts for clarity.")

            pos = nx.spring_layout(ego, seed=42, k=0.6)

            edge_x, edge_y = [], []
            for u, v in ego.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=0.9, color="rgba(0,217,192,0.25)"),
                hoverinfo="none",
            )

            node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
            for n in ego.nodes():
                x, y = pos[n]
                node_x.append(x)
                node_y.append(y)
                risk = float(score_of.get(n, 0.0))
                node_color.append(risk)
                is_center = (n == account_id)
                node_size.append(26 if is_center else 10 + risk * 14)
                fraud_flag = "FRAUD-LINKED" if label_of.get(n, 0) == 1 else "clean history"
                node_text.append(f"{n}<br>Risk score: {risk:.3f}<br>{fraud_flag}")

            # Soft "glow" layer behind the real markers -- same positions/colors,
            # larger and more transparent, to fake a bloom/glow effect.
            glow_trace = go.Scatter(
                x=node_x, y=node_y, mode="markers",
                marker=dict(size=[s * 2.2 for s in node_size], color=node_color,
                            colorscale="RdYlGn_r", cmin=0, cmax=1, opacity=0.18, line=dict(width=0)),
                hoverinfo="skip", showlegend=False,
            )
            node_trace = go.Scatter(
                x=node_x, y=node_y, mode="markers",
                marker=dict(size=node_size, color=node_color, colorscale="RdYlGn_r", cmin=0, cmax=1,
                            colorbar=dict(title="Risk score"), line=dict(width=1, color="white")),
                text=node_text, hoverinfo="text",
            )

            fig = go.Figure(data=[edge_trace, glow_trace, node_trace])
            fig.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=520,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                font=dict(family="Inter, sans-serif", color="#FAFAFA"),
            )
            st.plotly_chart(fig, width="stretch")

            st.info(
                f"**{account_id}** — risk score **{score_of[account_id]:.3f}** — "
                f"{'🚩 fraud-linked' if label_of[account_id] == 1 else '✅ no known fraud link'} — "
                f"{direct_links} direct connections"
            )

    with right:
        section_header("Triage", "Most Suspicious Fraud Rings")
        st.write("Communities detected via Louvain clustering, ranked by fraud density.")

        ring_df = rings.head(15).rename(columns={
            "size": "Size",
            "fraud_count": "Fraud accounts",
            "fraud_density": "Fraud density",
            "lead_account": "Highest-risk account",
        })[["Size", "Fraud accounts", "Fraud density", "Highest-risk account"]]
        ring_df["Fraud density"] = ring_df["Fraud density"] * 100
        st.dataframe(
            ring_df,
            column_config={
                "Fraud density": st.column_config.ProgressColumn(
                    "Fraud density", min_value=0, max_value=100, format="%.0f%%"
                ),
            },
            hide_index=True,
            width="stretch",
        )
        st.caption("Paste a highest-risk account into the search box to see its ring.")

        st.metric("Flagged rings", f"{results.get('num_flagged_communities', 0):,}")
        st.metric("Total communities found", f"{results.get('num_communities', 0):,}")


# ============================================================
# TAB 2: model comparison + the balance-leak feature importance story
# ============================================================
with tab_models:
    st.write("")
    section_header("Evaluate", "How each model performed")
    st.markdown('<p class="chart-note">Precision-Recall AUC on the held-out test set. PR-AUC suits '
                'rare-event problems like fraud, where accuracy is misleading.</p>', unsafe_allow_html=True)

    model_rows = [
        ("GraphSAGE alone (account-level)", "graphsage_pr_auc", "graph"),
        ("XGBoost tabular, naive", "tabular_pr_auc", "leak"),
        ("XGBoost hybrid, naive", "hybrid_pr_auc", "leak"),
        ("XGBoost tabular, hardened", "hardened_tabular_pr_auc", "fair"),
        ("XGBoost hybrid + embeddings, hardened", "hardened_hybrid_pr_auc", "fair"),
        ("XGBoost hybrid + risk score, hardened", "score_hybrid_pr_auc", "fair"),
    ]
    mdf = pd.DataFrame(
        [(name, results[key], kind) for name, key, kind in model_rows if key in results],
        columns=["model", "pr_auc", "kind"],
    )
    if not mdf.empty:
        colors = {"leak": MUTED, "fair": CLEAN, "graph": ACCENT}
        legend_names = {"leak": "Inflated by balance leak", "fair": "Fair comparison (leak removed)",
                        "graph": "Graph model alone"}
        fig = go.Figure()
        for kind in ["fair", "leak", "graph"]:
            part = mdf[mdf["kind"] == kind]
            if part.empty:
                continue
            fig.add_bar(
                y=part["model"], x=part["pr_auc"], orientation="h", name=legend_names[kind],
                marker=dict(color=colors[kind]),
                text=[f"{v:.3f}" for v in part["pr_auc"]], textposition="outside",
                textfont=dict(color=INK_SOFT),
                hovertemplate="%{y}<br>PR-AUC %{x:.4f}<extra></extra>",
            )
        base = results.get("overall_fraud_rate")
        if base:
            fig.add_vline(x=base, line=dict(color=INK_SOFT, dash="dot", width=1))
        fig.update_yaxes(categoryorder="array", categoryarray=mdf["model"].tolist()[::-1])
        fig.update_xaxes(range=[0, 1.12], title=None)
        style(fig, height=420)
        fig.update_layout(margin=dict(t=10, b=10),
                          legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0))
        st.plotly_chart(fig, width="stretch", theme=None)
        if base:
            st.markdown(f'<p class="chart-note">Dotted line: random guessing (PR-AUC ≈ fraud rate, '
                        f'{base:.3f}).</p>', unsafe_allow_html=True)

    st.write("")
    section_header("Diagnose", "Why the first model looked too good")
    c1, c2 = st.columns([3, 2])
    with c1:
        if fi is not None and not fi.empty:
            leak_cols = {"oldbalanceOrg", "newbalanceOrig"}
            naive = fi[fi["model"].str.startswith("Naive")].sort_values("importance")
            fig = go.Figure(go.Bar(
                y=naive["feature"], x=naive["importance"], orientation="h",
                marker=dict(color=[FRAUD if f in leak_cols else CLEAN for f in naive["feature"]]),
                hovertemplate="%{y}<br>importance %{x:.3f}<extra></extra>",
            ))
            fig.update_xaxes(title="Share of model's decisions", tickformat=".0%")
            st.plotly_chart(style(fig, height=340, legend=False), width="stretch", theme=None)
            st.caption("Coral bars: the two origin-balance columns that leak the label in PaySim.")
        else:
            st.info("Feature importances weren't exported. Re-run scripts/export_dashboard_data.py.")
    with c2:
        if fi is not None and not fi.empty:
            share = fi[(fi["model"].str.startswith("Naive")) &
                       (fi["feature"].isin(["oldbalanceOrg", "newbalanceOrig"]))]["importance"].sum()
            big = f"{share:.0%}"
        else:
            big = "68%"
        insight_card(
            "Data leak caught", "Two columns were doing the model's job",
            "The naive model leaned on the origin account's balance before and after each transaction. "
            "In the PaySim simulator, fraud drains the origin account to exactly zero, so these columns "
            "leak the answer. Removing them dropped PR-AUC from 0.996 to a credible 0.883.",
            big=big,
        )


# ============================================================
# TAB 3: what fraud looks like in the data
# ============================================================
@st.cache_data
def pattern_tables(_edges: pd.DataFrame):
    by_type = (_edges.groupby("type", observed=True)
               .agg(transactions=("is_fraud", "size"), fraud=("is_fraud", "sum")).reset_index())
    by_type["fraud_rate"] = by_type["fraud"] / by_type["transactions"]
    by_type = by_type.sort_values("fraud_rate", ascending=False)

    amt = _edges[_edges["amount"] > 0]
    bins = np.logspace(np.log10(max(amt["amount"].min(), 1)), np.log10(amt["amount"].max()), 45)
    centers = np.sqrt(bins[:-1] * bins[1:])
    dist = {}
    for flag, name in [(0, "Legitimate"), (1, "Fraud")]:
        counts, _ = np.histogram(amt.loc[amt["is_fraud"] == flag, "amount"], bins=bins)
        dist[name] = counts / max(counts.sum(), 1)

    daily = _edges.assign(day=_edges["step"] // 24).groupby("day")["is_fraud"].agg(["sum", "size"]).reset_index()
    return by_type, centers, dist, daily


with tab_patterns:
    st.write("")
    if "amount" not in edges.columns:
        st.info("Transaction details weren't exported. Re-run scripts/export_dashboard_data.py.")
    else:
        by_type, centers, dist, daily = pattern_tables(edges)
        c1, c2 = st.columns(2)
        with c1:
            section_header("Where", "Fraud rate by transaction type")
            fig = go.Figure(go.Bar(
                x=by_type["type"].astype(str), y=by_type["fraud_rate"],
                marker=dict(color=[FRAUD if r > 0 else MUTED for r in by_type["fraud_rate"]]),
                text=[f"{r:.1%}" for r in by_type["fraud_rate"]], textposition="outside",
                textfont=dict(color=INK_SOFT),
                customdata=np.stack([by_type["fraud"], by_type["transactions"]], axis=1),
                hovertemplate="%{x}<br>fraud rate %{y:.2%}<br>%{customdata[0]:,} of "
                              "%{customdata[1]:,} transactions<extra></extra>",
            ))
            fig.update_yaxes(tickformat=".0%", title="Share of transactions that are fraud")
            st.plotly_chart(style(fig, legend=False), width="stretch", theme=None)
            st.markdown('<p class="chart-note">PaySim fraud follows a two-step pattern: TRANSFER money out of a '
                        'victim account, then CASH_OUT from the receiving account.</p>', unsafe_allow_html=True)
        with c2:
            section_header("How much", "Transaction amounts: fraud vs legitimate")
            fig = go.Figure()
            for name, color, fill in [("Legitimate", CLEAN, "rgba(20,168,154,0.14)"),
                                      ("Fraud", FRAUD, "rgba(232,80,111,0.14)")]:
                fig.add_scatter(x=centers, y=dist[name], name=name, mode="lines",
                                line=dict(color=color, width=2, shape="spline"),
                                fill="tozeroy", fillcolor=fill,
                                hovertemplate=name + "<br>≈ %{x:,.0f}<br>%{y:.1%} of transactions<extra></extra>")
            fig.update_xaxes(type="log", title="Amount (log scale)")
            fig.update_yaxes(tickformat=".0%", title="Share within group")
            st.plotly_chart(style(fig), width="stretch", theme=None)
            st.markdown('<p class="chart-note">Each line shows how amounts are distributed within its own group, '
                        'so the shapes are comparable despite fraud being rare.</p>',
                        unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            section_header("When", "Fraud transactions per simulated day")
            fig = go.Figure(go.Scatter(
                x=daily["day"], y=daily["sum"], mode="lines", line=dict(color=FRAUD, width=2),
                hovertemplate="day %{x}<br>%{y:,} fraud transactions<extra></extra>",
            ))
            fig.update_xaxes(title="Day (PaySim step ÷ 24)")
            fig.update_yaxes(title="Fraud transactions")
            st.plotly_chart(style(fig, legend=False), width="stretch", theme=None)
        with c4:
            section_header("Graph signal", "GraphSAGE risk scores by account")
            fig = go.Figure()
            for flag, name, color in [(0, "Clean accounts", CLEAN), (1, "Fraud-linked accounts", FRAUD)]:
                counts, edges_ = np.histogram(nodes.loc[nodes["label"] == flag, "score"], bins=40, range=(0, 1))
                share = counts / max(counts.sum(), 1)
                fig.add_bar(x=(edges_[:-1] + edges_[1:]) / 2, y=share, name=name,
                            marker=dict(color=color), opacity=0.85,
                            hovertemplate=name + "<br>score ≈ %{x:.2f}<br>%{y:.1%} of group<extra></extra>")
            fig.update_layout(barmode="overlay", bargap=0.05)
            fig.update_xaxes(title="Risk score", range=[0, 1])
            fig.update_yaxes(tickformat=".0%", title="Share within group")
            st.plotly_chart(style(fig), width="stretch", theme=None)
            st.markdown('<p class="chart-note">The further apart these two distributions sit, the more signal the '
                        'graph model has learned from account neighborhoods.</p>', unsafe_allow_html=True)


# ============================================================
# TAB 4: ring detection
# ============================================================
with tab_rings:
    st.write("")
    k1, k2, k3 = st.columns(3)
    k1.metric("Candidate rings (size ≥ 3)", f"{len(comm):,}" if comm is not None else "–")
    k2.metric("Flagged as likely rings", f"{results.get('num_flagged_communities', 0):,}")
    k3.metric("Fraud-linked accounts captured", f"{results.get('ring_detection_recall', 0):.1%}")
    st.write("")
    section_header("Detect", "Every candidate ring: size vs fraud density")
    if comm is not None and not comm.empty:
        thr = results.get("flag_threshold", None)
        # Many rings share the same (size, density), so aggregate them into
        # bubbles sized by how many rings sit at each point.
        agg = (comm.groupby(["size", "fraud_density", "flagged"]).size()
               .rename("rings").reset_index())
        max_r = max(agg["rings"].max(), 1)
        fig = go.Figure()
        for flagged, name, color in [(False, "Not flagged", "rgba(250,250,250,0.30)"),
                                     (True, "Flagged ring", FRAUD)]:
            part = agg[agg["flagged"] == flagged]
            fig.add_trace(go.Scattergl(
                x=part["size"], y=part["fraud_density"], mode="markers", name=name,
                marker=dict(color=color, size=8 + 34 * np.sqrt(part["rings"] / max_r),
                            line=dict(width=1, color="#0a0e17"), opacity=0.85),
                customdata=part["rings"],
                hovertemplate="size %{x} accounts<br>fraud density %{y:.1%}<br>"
                              "%{customdata:,} rings here<extra></extra>",
            ))
        if thr:
            fig.add_hline(y=thr, line=dict(color=INK_SOFT, dash="dot", width=1))
            fig.add_annotation(x=1, xref="paper", y=thr, text=f"flag threshold {thr:.1%} (5× average)",
                               showarrow=False, xanchor="right", yanchor="bottom",
                               font=dict(color=INK_SOFT, size=11))
        ticks = [t for t in [3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000] if t <= agg["size"].max() * 1.5]
        fig.update_xaxes(type="log", title="Ring size (accounts, log scale)", tickvals=ticks,
                         ticktext=[f"{t:,}" for t in ticks])
        fig.update_yaxes(tickformat=".0%", title="Fraud density", range=[-0.02, 1.05])
        st.plotly_chart(style(fig, height=420), width="stretch", theme=None)
        st.markdown('<p class="chart-note">Louvain groups accounts that transact densely with each other. '
                    'A ring is flagged when its share of fraud-linked accounts is at least five times the '
                    'graph-wide average. Bubble size shows how many rings share that point.</p>', unsafe_allow_html=True)
    else:
        st.info("Ring summary wasn't exported. Re-run scripts/export_dashboard_data.py.")


# ============================================================
# TAB 5: key findings
# ============================================================
with tab_findings:
    st.write("")
    section_header("Takeaways", "What this project found")
    a, b, c = st.columns(3)
    with a:
        insight_card(
            "Finding 1", "A near-perfect score was a data leak",
            "The first model scored 0.996 PR-AUC. Feature importance showed it relied on a PaySim "
            "simulator artifact, not real fraud behavior. Removing the leaking columns gave an honest "
            f"{results.get('hardened_tabular_pr_auc', 0.8832):.3f}.",
            big=f"{results.get('tabular_pr_auc', 0.9962):.3f} → {results.get('hardened_tabular_pr_auc', 0.8832):.3f}",
        )
    with b:
        insight_card(
            "Finding 2", "Graph features didn't lift per-transaction scoring",
            "Adding GraphSAGE embeddings or risk scores to XGBoost didn't beat tabular features alone. "
            "PaySim fraud is mostly a short two-step chain, so there's little multi-hop structure "
            "for the graph to add at the single-transaction level.",
            big=f"{results.get('score_hybrid_pr_auc', 0.8817):.3f} vs {results.get('hardened_tabular_pr_auc', 0.8832):.3f}",
        )
    with c:
        insight_card(
            "Finding 3", "The graph's real value is finding rings",
            f"Louvain community detection flagged {results.get('num_flagged_communities', 0):,} dense clusters "
            f"capturing {results.get('ring_detection_recall', 0):.1%} of fraud-linked accounts, something a "
            "model that scores one transaction at a time has no way to see.",
            big=f"{results.get('num_flagged_communities', 0):,} rings",
        )
    st.write("")
    with st.expander("Limitations and what a production version would add"):
        st.markdown(
            "- **Synthetic data.** PaySim is simulated; real fraud includes shared devices, IPs and longer "
            "multi-hop rings, where graph features would likely add more.\n"
            "- **Sampled graph.** Built from all fraud plus 300,000 legitimate transactions, not the full 6.3M rows.\n"
            "- **Batch, not streaming.** A real system would ingest transactions continuously (for example, Kafka) "
            "and use a graph database such as Neo4j.\n"
            "- **Adversaries adapt.** Fraud rings change behavior once detected, so the model needs monitoring "
            "and regular retraining."
        )
