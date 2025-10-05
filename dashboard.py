import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Streamlit page setup
st.set_page_config(page_title="AI Model Comparison Dashboard", layout="wide")
st.title("🤖 AI Model Evaluation Dashboard")

# Step 1: Load Excel files
excel_files = [f for f in os.listdir() if f.endswith(('.xls', '.xlsx'))]

if not excel_files:
    st.error("❌ No Excel files found in directory.")
    st.stop()

dataframes = {}
for file in excel_files:
    try:
        if file.endswith('.xls'):
            df = pd.read_excel(file, engine='xlrd')
        else:
            df = pd.read_excel(file)
        dataframes[file] = df
    except Exception as e:
        st.warning(f"⚠️ Could not read {file}: {e}")

# Step 2: Select file
selected_file = st.selectbox("Select a dataset to analyze:", list(dataframes.keys()))
df = dataframes[selected_file]

st.write("### Data Preview")
st.dataframe(df.head())

# Step 3: Check required columns
required_cols = [
    "Prompt_Type",
    "Model",
    "Answer quality (1-5, 0 if it's wrong)",
    "Electricity consumption",
    "CO2 emission",
    "Inference timing (seconds)"
]

missing = [col for col in required_cols if col not in df.columns]
if missing:
    st.error(f"❌ Missing columns: {missing}")
    st.stop()

# --- Step 4: Visualizations ---
st.markdown("---")
st.subheader("📈 Visualizations")

sns.set_theme(style="whitegrid")

# --- Ensure correct dtypes before plotting ---
df["Prompt_Type"] = df["Prompt_Type"].astype(str)
df["Model"] = df["Model"].astype(str)

numeric_cols = [
    "Answer quality (1-5, 0 if it's wrong)",
    "Electricity consumption",
    "CO2 emission",
    "Inference timing (seconds)"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# --- Common styling function ---
def clean_axis(ax, xlabel="", ylabel=""):
    ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=8)
    ax.tick_params(axis='x', labelrotation=30, labelsize=8, pad=4)
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(title="Model", fontsize=8, title_fontsize=9, frameon=False, loc='upper right')
    plt.tight_layout()

# --- Helper to make consistent plots ---
def make_barplot(data, x, y, ylabel):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=data,
        x=x,
        y=y,
        hue="Model",
        ax=ax,
        palette="Set2",
        dodge=True,
        errorbar=None
    )
    clean_axis(ax, xlabel="Prompt Type", ylabel=ylabel)
    st.pyplot(fig)

# --- Graphs ---
st.write("### 1️⃣ Answer Quality vs Energy Consumption")
make_barplot(df, "Prompt_Type", "Electricity consumption", "Electricity Consumption (kWh)")

st.write("### 2️⃣ Answer Quality vs CO2 Emission (Cost Proxy)")
make_barplot(df, "Prompt_Type", "CO2 emission", "CO2 Emission (kg)")

st.write("### 3️⃣ Inference Latency Comparison")
make_barplot(df, "Prompt_Type", "Inference timing (seconds)", "Inference Time (seconds)")

# --- Trade-off Table ---
st.write("### 4️⃣ Best Trade-off Summary Table")

best_tradeoff = (
    df.groupby(["Prompt_Type", "Model"], dropna=False)
    .agg({
        "Answer quality (1-5, 0 if it's wrong)": "mean",
        "Electricity consumption": "mean",
        "CO2 emission": "mean",
        "Inference timing (seconds)": "mean"
    })
    .reset_index()
)

best_tradeoff = best_tradeoff.round(3)
st.dataframe(best_tradeoff)

st.success("✅ Dashboard visuals fixed — no more type or overlap errors!")
