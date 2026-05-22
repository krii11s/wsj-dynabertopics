from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

BASE = Path(__file__).parent / "Nomic" / "v2_forzen2"

# ---- Load and merge data (cached so re-runs don't re-read CSVs) ----
@st.cache_data
def load_data():
    att = pd.read_csv(BASE / "topic_attention.csv")
    lab = pd.read_csv(BASE / "topic_llm_labels.csv")
    df = att.merge(lab[["topic_id", "label", "description"]],
                   on="topic_id", how="left")
    df["label"] = df["label"].fillna("(unlabeled)")
    df["display"] = "T" + df["topic_id"].astype(str) + " — " + df["label"]
    df["month_dt"] = pd.to_datetime(df["month"])
    return df

df = load_data()

st.set_page_config(page_title="WSJ Topics (v2_forzen2)", layout="wide")
st.title("WSJ Topic Trajectory Explorer — v2_forzen2")
st.caption("Dynamic BERTopic model on 22 years of Wall Street Journal articles (2000–2021).")

# ---- Sidebar: search + multiselect ----
all_topics = (df.drop_duplicates("topic_id")
                .sort_values("topic_id")[["topic_id", "display"]])

with st.sidebar:
    st.header("Topic selection")
    keyword = st.text_input("Filter by keyword (matches label)", "")
    pool = all_topics[all_topics["display"].str.contains(keyword,
                                                         case=False, na=False)]
    picked = st.multiselect(
        f"Pick topics to plot ({len(pool)} available)",
        options=pool["display"].tolist(),
        default=pool["display"].head(3).tolist(),  # default: first 3
    )
    metric = st.radio(
        "Y-axis metric",
        ["attention", "n_topic_docs"],
        format_func=lambda x: "Attention share" if x == "attention"
                              else "Monthly article count",
    )

if not picked:
    st.info("Pick at least one topic from the sidebar.")
    st.stop()

sub = df[df["display"].isin(picked)]

# ---- Line chart ----
fig = px.line(
    sub.sort_values("month_dt"),
    x="month_dt", y=metric, color="display",
    hover_data=["topic_id", "n_topic_docs", "n_month_docs", "attention"],
)
y_label = "Attention share (topic docs / month total)" \
          if metric == "attention" else "Monthly article count"
fig.update_layout(
    height=520,
    xaxis_title="Month",
    yaxis_title=y_label,
    legend_title="Topic",
)
st.plotly_chart(fig, use_container_width=True)

# ---- Download button ----
st.download_button(
    "Download selected data (CSV)",
    sub.to_csv(index=False).encode("utf-8-sig"),
    "selected_topics.csv",
    "text/csv",
)

# ---- Topic descriptions ----
with st.expander("LLM descriptions of selected topics"):
    desc = (sub.drop_duplicates("topic_id")[["topic_id", "label", "description"]]
              .sort_values("topic_id"))
    st.dataframe(desc, use_container_width=True, hide_index=True)
