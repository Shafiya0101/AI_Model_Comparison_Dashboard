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

# Step 2: Select file to display
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

# Step 4: Visualizations
st.markdown("---")
st.subheader("📈 Visualizations")

sns.set_theme(style="whitegrid")

# Common styling function
def clean_axis(ax):
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_fontsize(9)
    ax.set_xlabel("Prompt Type", fontsize=10)
    ax.set_ylabel(ax.get_ylabel(), fontsize=10)
    ax.tick_params(axis='x', pad=6)
    plt.tight_layout()

# Graph 1: Quality vs Energy
st.write("### 1️⃣ Answer Quality vs Energy Consumption")
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(
    data=df,
    x="Prompt_Type",
    y="Electricity consumption",
    hue="Model",
    ax=ax
)
ax.set_ylabel("Electricity Consumption (kWh)")
clean_axis(ax)
st.pyplot(fig)

# Graph 2: Quality vs Cost (CO2 emission)
st.write("### 2️⃣ Answer Quality vs CO2 Emission (Cost Proxy)")
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(
    data=df,
    x="Prompt_Type",
    y="CO2 emission",
    hue="Model",
    ax=ax
)
ax.set_ylabel("CO2 Emission (kg)")
clean_axis(ax)
st.pyplot(fig)

# Graph 3: Latency Comparison
st.write("### 3️⃣ Inference Latency Comparison")
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(
    data=df,
    x="Prompt_Type",
    y="Inference timing (seconds)",
    hue="Model",
    ax=ax
)
ax.set_ylabel("Inference Time (s)")
clean_axis(ax)
st.pyplot(fig)

# Graph 4: Best Trade-off Table
st.write("### 4️⃣ Best Trade-off Summary Table")
best_tradeoff = (
    df.groupby(["Prompt_Type", "Model"])
    .agg({
        "Answer quality (1-5, 0 if it's wrong)": "mean",
        "Electricity consumption": "mean",
        "CO2 emission": "mean",
        "Inference timing (seconds)": "mean"
    })
    .reset_index()
)
st.dataframe(best_tradeoff)

st.success("✅ Dashboard ready with improved, readable charts!")
