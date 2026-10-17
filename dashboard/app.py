"""
Step 12: FedShield-IDS dashboard (Streamlit).

Displays the tables written by experiments/summarize_results.py
(results/summary/). Every number on screen comes from those CSVs (the
dashboard only reshapes them: averages across clients, pivots for heatmaps),
so it cannot disagree with the report. The caveats shown under each chart
are read from summary_report.md.

Run from the project root:
    streamlit run dashboard/app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DEFAULT_DIR = "results/summary"
METRIC_LABELS = {"macro_f1": "macro-F1", "accuracy": "accuracy"}
PALETTE = ["#1E78E6", "#F28C28", "#2BA84A", "#8E44AD", "#D64545", "#17A2B8", "#7F8C8D"]
INK = "#0B2545"

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = PALETTE

st.set_page_config(page_title="FedShield-IDS", layout="wide")

st.markdown("""
<style>
.fs-banner {background: linear-gradient(90deg, #0B3C8C 0%, #1E78E6 60%, #5AB0FF 100%);
            padding: 1.3rem 1.8rem; border-radius: 14px; margin-bottom: 1rem;}
.fs-banner h1 {color: #FFFFFF !important; margin: 0; padding: 0; font-size: 2rem;}
.fs-banner p {color: #DCEBFF; margin: .3rem 0 0 0; font-size: 1.05rem;}
h2, h3 {color: #1565C0 !important;}
h3 {border-bottom: 2px solid #BBD7F5; padding-bottom: .25rem;}
[data-testid="stMetric"] {background: #E6F1FF; border-left: 5px solid #1E78E6;
                          padding: .8rem 1rem; border-radius: 10px;}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {color: #0B2545 !important;}
button[data-baseweb="tab"][aria-selected="true"] {color: #1565C0 !important; font-weight: 600;}
div[data-baseweb="tab-highlight"] {background-color: #1E78E6 !important;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Loading and helpers
# ----------------------------------------------------------------------

def parse_notes(report_text: str, stems: list) -> dict:
    """Pull the italic caveat under each '## title' in the report, matched to tables by order."""
    sections = []
    for line in report_text.splitlines():
        if line.startswith("## "):
            sections.append([line[3:], ""])
        elif sections and not sections[-1][1] and line.startswith("_") and line.endswith("_"):
            sections[-1][1] = line[1:-1]
    if len(sections) != len(stems):  # report and tables out of step: show no notes rather than wrong ones
        return {}
    return {stem: note for stem, (_, note) in zip(stems, sections)}


@st.cache_data
def load(summary_dir: str):
    d = Path(summary_dir)
    tables = {p.stem: pd.read_csv(p) for p in sorted(d.glob("*.csv"))}
    report_path = d / "summary_report.md"
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    return tables, parse_notes(report, list(tables)), report


def wide_to_long(df: pd.DataFrame, prefixes: list, metric: str) -> pd.DataFrame:
    """Turn columns like baseline_macro_f1 / secagg_macro_f1 into one tidy table."""
    parts = []
    for p in prefixes:
        col = f"{p}{metric}"
        if col in df.columns:
            parts.append(pd.DataFrame({"client": df["client"],
                                       "run": p.rstrip("_").replace("_", " "),
                                       metric: df[col]}))
    return pd.concat(parts, ignore_index=True)


def last3_grid(curves: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Run x client table of each run's mean over its final 3 rounds."""
    rows = {}
    for run, g in curves.groupby("run", sort=False):
        last = g["round"].max()
        rows[run] = g[g["round"] > last - 3].groupby("client")[metric].mean()
    return pd.DataFrame(rows).T


def show(fig):
    """One place that applies the look to every chart."""
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF", font=dict(color=INK),
                      legend_title_text="", margin=dict(t=40, b=40))
    fig.update_xaxes(gridcolor="#E3ECF7")
    fig.update_yaxes(gridcolor="#E3ECF7")
    st.plotly_chart(fig, theme=None)  # theme=None keeps our palette instead of Streamlit's


def grouped_bars(df, x, y, color, **kwargs):
    return px.bar(df, x=x, y=y, color=color, barmode="group", **kwargs)


def roc_figure(d: pd.DataFrame, curve_col: str):
    fig = px.line(d, x="fpr", y="tpr", color=curve_col, range_x=[0, 1], range_y=[0, 1.02])
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                             line=dict(dash="dash", color="#7F8C8D")))
    fig.update_layout(xaxis_title="false positive rate", yaxis_title="true positive rate")
    return fig


def label_with_auc(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["curve"] = d["model"] + " (AUC " + d["auc"].map("{:.3f}".format) + ")"
    return d


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

st.sidebar.title("FedShield-IDS")
summary_dir = st.sidebar.text_input("Summary folder", DEFAULT_DIR)
metric = st.sidebar.radio("Per-client metric", list(METRIC_LABELS), format_func=METRIC_LABELS.get)
st.sidebar.caption("Accuracy can look high for a model that predicts one class for everything. "
                   "Macro-F1 averages over classes, so it can't be gamed that way.")
if st.sidebar.button("Reload tables"):
    st.cache_data.clear()

if not Path(summary_dir).is_dir() or not list(Path(summary_dir).glob("*.csv")):
    st.error(f"No summary tables found in '{summary_dir}'. From the project root, run:\n\n"
             "`python experiments/summarize_results.py`")
    st.stop()

tables, notes, report = load(summary_dir)
ylab = METRIC_LABELS[metric]


def note(stem: str):
    if notes.get(stem):
        st.caption(notes[stem])


def show_table(stem: str):
    with st.expander("Show table"):
        st.dataframe(tables[stem], hide_index=True)


st.markdown('<div class="fs-banner"><h1>FedShield-IDS</h1>'
            '<p>Privacy-preserving federated intrusion detection across five organizations</p></div>',
            unsafe_allow_html=True)

NAMES = ["Overview", "Data heterogeneity", "Baseline", "Comparison", "Secure aggregation",
         "Robustness (Krum)", "Differential privacy", "Dropout", "ROC curves",
         "Membership attack", "Full report"]
tab = dict(zip(NAMES, st.tabs(NAMES)))

# ----------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------
with tab["Overview"]:
    st.subheader("Headline numbers")
    c1, c2, c3, c4 = st.columns(4)

    o = tables.get("03b_secagg_overhead")
    if o is not None and (o["run"] == "secagg").any():
        c1.metric("Secure aggregation overhead",
                  f"{o.loc[o['run'] == 'secagg', 'overhead_pct'].iloc[0]:.2f}%",
                  help="Extra wall-clock time versus plain FedAvg, same 15 rounds.")

    k = tables.get("04b_krum_selection")
    if k is not None and k["is_attacker"].any():
        a = k[k["is_attacker"]].iloc[0]
        c2.metric("Krum picked the attacker", f"{int(a['rounds_selected'])} of {int(a['rounds_total'])} rounds")

    e = tables.get("06_dp_ecommerce")
    if e is not None and "utility_vs_plain" in e:
        dp = e[np.isfinite(e["epsilon"])]
        if len(dp):
            c3.metric("Max accuracy cost of DP (E-commerce)",
                      f"{(1 - dp['utility_vs_plain'].min()) * 100:.1f}%",
                      help=f"Balanced accuracy, across epsilon {dp['epsilon'].min():.2f} to "
                           f"{dp['epsilon'].max():.2f}.")

    aucs = [tables[s]["auc"] for s in ("08_mia", "09_mia_stress") if s in tables]
    if aucs:
        all_auc = pd.concat(aucs)
        c4.metric("Highest membership-attack AUC", f"{all_auc.max():.3f}",
                  help=f"Across {len(all_auc)} models. 0.5 means the attacker is guessing.")

    st.write("Use the tabs above for each experiment. The last tab holds the full written report, "
             "including the caveats for every table.")

# ----------------------------------------------------------------------
# Data heterogeneity
# ----------------------------------------------------------------------
with tab["Data heterogeneity"]:
    if "01_noniid_divergence" in tables:
        d = tables["01_noniid_divergence"].sort_values("js_divergence", ascending=False)
        st.subheader("How different each client's data is")
        fig = px.bar(d, x="client", y="js_divergence", text_auto=".3f",
                     hover_data=["n_rows", "n_classes_present"])
        fig.update_layout(yaxis_title="Jensen-Shannon divergence from pooled data")
        show(fig)
        note("01_noniid_divergence")
        show_table("01_noniid_divergence")
    if "00_class_distribution" in tables:
        st.subheader("Pooled class mix")
        show(px.bar(tables["00_class_distribution"], x="class", y="proportion", text_auto=".3f"))

# ----------------------------------------------------------------------
# Baseline
# ----------------------------------------------------------------------
with tab["Baseline"]:
    if "02_weighting_ablation" in tables:
        st.subheader("How client weighting changed the baseline")
        fig = grouped_bars(tables["02_weighting_ablation"], "client", metric, "run",
                           hover_data=["rounds"], range_y=[0, 1])
        fig.update_layout(yaxis_title=ylab)
        show(fig)
        note("02_weighting_ablation")
        show_table("02_weighting_ablation")

# ----------------------------------------------------------------------
# Comparison (line charts + heatmap)
# ----------------------------------------------------------------------
with tab["Comparison"]:
    curves = tables.get("10_learning_curves")
    if curves is None:
        st.info("No learning-curve table yet. Run `python experiments/summarize_results.py`.")
    else:
        st.subheader("Learning curves: compare runs round by round")
        runs = list(dict.fromkeys(curves["run"]))
        default = [r for r in ("baseline (final)", "secure aggregation", "client dropout") if r in runs]
        left, right = st.columns([2, 1])
        chosen = left.multiselect("Runs to compare", runs, default=default or runs[:3])
        view = right.selectbox("Show", ["Average across clients"] + sorted(curves["client"].unique()))
        d = curves[curves["run"].isin(chosen)]
        if view == "Average across clients":
            d = d.groupby(["run", "round"], as_index=False)[metric].mean()
        else:
            d = d[d["client"] == view]
        fig = px.line(d, x="round", y=metric, color="run", markers=True, range_y=[0, 1])
        fig.update_layout(yaxis_title=ylab)
        show(fig)
        note("10_learning_curves")

        st.subheader("Heatmap: every run, every client (mean of the last 3 rounds)")
        fig = px.imshow(last3_grid(curves, metric), text_auto=".2f", color_continuous_scale="Blues",
                        zmin=0, zmax=1, aspect="auto")
        fig.update_layout(coloraxis_colorbar_title=ylab, xaxis_title="client", yaxis_title="")
        show(fig)

# ----------------------------------------------------------------------
# Secure aggregation
# ----------------------------------------------------------------------
with tab["Secure aggregation"]:
    if "03_secagg" in tables:
        st.subheader("Secure aggregation vs plain FedAvg")
        long = wide_to_long(tables["03_secagg"], ["baseline_", "secagg_"], metric)
        fig = grouped_bars(long, "client", metric, "run", range_y=[0, 1])
        fig.update_layout(yaxis_title=ylab)
        show(fig)
        note("03_secagg")
        show_table("03_secagg")
    if "03b_secagg_overhead" in tables:
        st.subheader("Runtime")
        st.dataframe(tables["03b_secagg_overhead"], hide_index=True)
        note("03b_secagg_overhead")

# ----------------------------------------------------------------------
# Krum
# ----------------------------------------------------------------------
with tab["Robustness (Krum)"]:
    if "04_krum_comparison" in tables:
        st.subheader("One malicious client: plain FedAvg vs Krum (final round)")
        long = wide_to_long(tables["04_krum_comparison"], ["fedavg_attacked_", "krum_defended_"], metric)
        fig = grouped_bars(long, "client", metric, "run", range_y=[0, 1])
        fig.update_layout(yaxis_title=ylab)
        show(fig)
        note("04_krum_comparison")
    if "04b_krum_selection" in tables:
        st.subheader("Whose update Krum kept")
        fig = px.bar(tables["04b_krum_selection"], x="client", y="rounds_selected", color="is_attacker",
                     color_discrete_map={True: "#D64545", False: "#1E78E6"}, text="rounds_selected")
        fig.update_layout(yaxis_title="rounds selected")
        show(fig)
        note("04b_krum_selection")

# ----------------------------------------------------------------------
# Differential privacy
# ----------------------------------------------------------------------
with tab["Differential privacy"]:
    if "06_dp_ecommerce" in tables:
        st.subheader("Privacy vs accuracy on E-commerce")
        e = tables["06_dp_ecommerce"]
        dp = e[np.isfinite(e["epsilon"])].sort_values("epsilon")
        plain = e[e["setting"] == "plain"]
        if len(dp):
            fig = px.line(dp, x="epsilon", y="balanced_accuracy", markers=True, log_x=True,
                          hover_data=["setting"])
            if len(plain):
                fig.add_hline(y=float(plain["balanced_accuracy"].iloc[0]), line_dash="dash",
                              line_color="#F28C28", annotation_text="plain training (no privacy)")
            lo = min(0.5, float(dp["balanced_accuracy"].min()) - 0.02)
            fig.update_yaxes(range=[lo, 1.0], title="balanced accuracy")
            fig.update_xaxes(title="epsilon (lower = stronger privacy)")
            show(fig)
        note("06_dp_ecommerce")
        show_table("06_dp_ecommerce")
    if "05_dp_federated" in tables:
        st.subheader("DP across all five clients")
        f = tables["05_dp_federated"].copy()
        f["run"] = ("noise " + f["noise_multiplier"].astype(str) + ", batch " +
                    f["batch_size"].astype(str))
        fig = grouped_bars(f, "client", "epsilon", "run", log_y=True)
        fig.update_layout(yaxis_title="epsilon (log scale)")
        show(fig)
        fig = grouped_bars(f, "client", metric, "run", range_y=[0, 1])
        fig.update_layout(yaxis_title=ylab)
        show(fig)
        note("05_dp_federated")
        show_table("05_dp_federated")

# ----------------------------------------------------------------------
# Dropout
# ----------------------------------------------------------------------
with tab["Dropout"]:
    if "07_dropout" in tables:
        st.subheader("Training with clients dropping out")
        d = tables["07_dropout"]
        long = wide_to_long(d, ["baseline_", "dropout_"], metric)
        fig = grouped_bars(long, "client", metric, "run", range_y=[0, 1])
        fig.update_layout(yaxis_title=ylab)
        show(fig)
        fig = px.bar(d, x="client", y="rounds_trained", text="rounds_trained")
        fig.update_layout(yaxis_title=f"rounds trained (of {int(d['rounds_total'].iloc[0])})")
        show(fig)
        note("07_dropout")
        show_table("07_dropout")

# ----------------------------------------------------------------------
# ROC curves
# ----------------------------------------------------------------------
with tab["ROC curves"]:
    det = tables.get("11_roc_detection")
    if det is None:
        st.info("No ROC data yet. Run `python experiments/roc_curves.py`, then "
                "`python experiments/summarize_results.py`.")
    else:
        st.subheader("Can the models tell attacks from normal traffic?")
        client = st.selectbox("Client", sorted(det["client"].unique()))
        show(roc_figure(label_with_auc(det[det["client"] == client]), "curve"))
        note("11_roc_detection")

        st.subheader("AUC by model and client")
        grid = det.groupby(["model", "client"], sort=False)["auc"].first().unstack()
        fig = px.imshow(grid, text_auto=".3f", color_continuous_scale="Blues", zmin=0.4, zmax=1.0,
                        aspect="auto")
        fig.update_layout(coloraxis_colorbar_title="AUC", xaxis_title="client", yaxis_title="")
        show(fig)

    mia_roc = tables.get("12_roc_mia")
    if mia_roc is not None:
        st.subheader("Membership attack: can an attacker tell training records from unseen ones?")
        show(roc_figure(label_with_auc(mia_roc), "curve"))
        note("12_roc_mia")

# ----------------------------------------------------------------------
# Membership attack
# ----------------------------------------------------------------------
with tab["Membership attack"]:
    def auc_chart(df, label_col, title):
        fig = go.Figure(go.Scatter(
            x=df[label_col], y=df["auc"], mode="markers", marker=dict(color=PALETTE[0], size=10),
            error_y=dict(type="data", symmetric=False, array=df["auc_hi"] - df["auc"],
                         arrayminus=df["auc"] - df["auc_lo"], color=PALETTE[0])))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#7F8C8D",
                      annotation_text="chance (attacker guessing)")
        fig.update_layout(title=title, yaxis_title="attack AUC (bars = 95% interval)",
                          yaxis_range=[0.3, 1.0])
        return fig

    if "08_mia" in tables:
        st.subheader("Can an attacker tell which records a model trained on?")
        show(auc_chart(tables["08_mia"], "model", "E-commerce models"))
        note("08_mia")
        show_table("08_mia")
    if "09_mia_stress" in tables:
        s = tables["09_mia_stress"].copy()
        s["setting"] = ("vectors " + s["vectors"].astype(str) + ", hidden " + s["hidden"].astype(str) +
                        ", epochs " + s["epochs"].astype(str))
        show(auc_chart(s, "setting", "Stress test: few training records, big models"))
        note("09_mia_stress")
        show_table("09_mia_stress")

# ----------------------------------------------------------------------
# Full report
# ----------------------------------------------------------------------
with tab["Full report"]:
    st.markdown(report if report else "_summary_report.md not found._")