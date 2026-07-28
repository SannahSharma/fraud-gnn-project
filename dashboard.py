"""
dashboard.py

Purpose: An interactive Streamlit dashboard for exploring the fraud
detection system -- headline metrics, a searchable graph neighborhood
explorer (visualizing WHY an account is flagged, by showing who it's
connected to), and a table of the most suspicious detected fraud rings.

Run with: streamlit run dashboard.py
"""

import pickle
import json
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


def section_header(kicker: str, title: str):
    """Render a small uppercase 'kicker' label above a bold section title --
    an editorial-style heading, used instead of plain st.subheader."""
    st.markdown(f'<div class="kicker">{kicker}</div><div class="section-title">{title}</div>',
                unsafe_allow_html=True)


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
@st.cache_resource
def load_data():
    with open("data/transaction_graph.pkl", "rb") as f:
        G = pickle.load(f)
    with open("data/graph_data.pkl", "rb") as f:
        graph_data = pickle.load(f)
    with open("data/node_scores.pkl", "rb") as f:
        node_score_data = pickle.load(f)
    with open("data/community_results.pkl", "rb") as f:
        community_data = pickle.load(f)
    with open("data/results.json", "r") as f:
        results = json.load(f)
    return G, graph_data, node_score_data, community_data, results

G, graph_data, node_score_data, community_data, results = load_data()

node_to_idx = graph_data["node_to_idx"]
labels = graph_data["data"].y.numpy()
node_scores = node_score_data["node_scores"]

# ============================================================
# KPI row
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accounts (nodes)", f"{G.number_of_nodes():,}")
col2.metric("Transactions (edges)", f"{G.number_of_edges():,}")
col3.metric("Tabular-only PR-AUC", f"{results.get('hardened_tabular_pr_auc', 0):.3f}")
col4.metric("Hybrid PR-AUC", f"{results.get('score_hybrid_pr_auc', 0):.3f}")
col5.metric("Ring-detection recall", f"{results.get('ring_detection_recall', 0) * 100:.1f}%")

st.write("")
st.write("")

# ============================================================
# Layout: neighborhood explorer (left) + suspicious rings table (right)
# ============================================================
left, right = st.columns([2, 1])

with left:
    section_header("Investigate", "Account Neighborhood Explorer")
    st.write("Search any account ID to see its 2-hop transaction neighborhood, "
             "colored by GraphSAGE fraud risk score.")

    top_risk_idx = np.argsort(-node_scores)[:25]
    idx_to_node = {i: n for n, i in node_to_idx.items()}
    suggested_accounts = [idx_to_node[i] for i in top_risk_idx if i in idx_to_node]

    account_id = st.selectbox(
        "Choose a high-risk account, or type any account ID below:",
        options=suggested_accounts,
        index=0,
    )
    custom_id = st.text_input("...or search for a specific account ID:")
    if custom_id.strip():
        account_id = custom_id.strip()

    if account_id not in node_to_idx:
        st.error(f"Account '{account_id}' was not found in the graph.")
    else:
        undirected_G = G.to_undirected()
        ego = nx.ego_graph(undirected_G, account_id, radius=2)

        MAX_NODES = 120
        if ego.number_of_nodes() > MAX_NODES:
            neighbor_scores = [
                (n, node_scores[node_to_idx[n]]) for n in ego.nodes() if n != account_id
            ]
            neighbor_scores.sort(key=lambda x: x[1], reverse=True)
            keep = {account_id} | {n for n, _ in neighbor_scores[:MAX_NODES - 1]}
            ego = ego.subgraph(keep)
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
            risk = node_scores[node_to_idx[n]] if n in node_to_idx else 0.0
            node_color.append(risk)
            is_center = (n == account_id)
            node_size.append(26 if is_center else 10 + risk * 14)
            fraud_flag = "FRAUD-LINKED" if labels[node_to_idx[n]] == 1 else "clean history"
            node_text.append(f"{n}<br>Risk score: {risk:.3f}<br>{fraud_flag}")

        # Soft "glow" layer behind the real markers -- same positions/colors,
        # larger and more transparent, to fake a bloom/glow effect.
        glow_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers",
            marker=dict(
                size=[s * 2.2 for s in node_size],
                color=node_color, colorscale="RdYlGn_r", cmin=0, cmax=1,
                opacity=0.18, line=dict(width=0),
            ),
            hoverinfo="skip", showlegend=False,
        )

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers",
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale="RdYlGn_r",
                cmin=0, cmax=1,
                colorbar=dict(title="Risk score"),
                line=dict(width=1, color="white"),
            ),
            text=node_text,
            hoverinfo="text",
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
        st.plotly_chart(fig, use_container_width=True)

        idx = node_to_idx[account_id]
        st.info(
            f"**{account_id}** — risk score **{node_scores[idx]:.3f}** — "
            f"{'🚩 fraud-linked' if labels[idx] == 1 else '✅ no known fraud link'} — "
            f"{ego.degree[account_id]} connections shown"
        )

with right:
    section_header("Triage", "Most Suspicious Fraud Rings")
    st.write("Communities detected via Louvain clustering, ranked by fraud density.")

    sizeable = [c for c in community_data["communities"] if c["size"] >= 3]
    sizeable.sort(key=lambda c: c["fraud_density"], reverse=True)

    table_rows = []
    for c in sizeable[:15]:
        table_rows.append({
            "Size": c["size"],
            "Fraud accounts": c["fraud_count"],
            "Fraud density": c["fraud_density"],
            "Sample account": next(iter(c["nodes"])),
        })

    ring_df = pd.DataFrame(table_rows)
    st.dataframe(
        ring_df,
        column_config={
            "Fraud density": st.column_config.ProgressColumn(
                "Fraud density", min_value=0, max_value=1, format="%.0f%%"
            ),
        },
        hide_index=True,
        use_container_width=True,
    )

    st.metric("Flagged rings", results.get("num_flagged_communities", 0))
    st.metric("Total communities found", results.get("num_communities", 0))