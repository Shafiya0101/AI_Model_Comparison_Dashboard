# 🤖 AI Model Comparison Dashboard

An interactive **Streamlit** dashboard for comparing how different AI models respond to prompts — weighing **answer quality** against the **energy, carbon, and latency** cost of producing each answer.

It loads one or more Excel result files, lets you switch between them, and visualizes the trade-offs so you can see which model gives the best answers for the least resource cost.

---

## 📊 What it does

- **Auto-loads** every `.xls` / `.xlsx` file in the project folder and lets you pick which dataset to analyze.
- **Data preview** of the selected file.
- **Validation** that the required columns are present before plotting.
- Three grouped bar charts, broken down by prompt type and model:
  1. **Electricity consumption** per prompt type
  2. **CO₂ emission** (cost proxy) per prompt type
  3. **Inference latency** (seconds) per prompt type
- A **best trade-off summary table** with the mean answer quality, electricity, CO₂, and inference time for each prompt type / model combination.

## 🗂️ Data format

Each Excel file should contain these columns:

| Prompt_Type | Model | Answer quality (1-5, 0 if it's wrong) | Electricity consumption | CO2 emission | Inference timing (seconds) |
|-------------|-------|----------------------------------------|-------------------------|--------------|----------------------------|

- **Answer quality** is rated 1–5, with 0 meaning the answer was wrong.
- Multiple files (e.g. `file1.xls`, `file2.xlsx`, `file3.xlsx`) can each hold a different run or model set.

## 🛠️ Tech stack

- Python
- [Streamlit](https://streamlit.io/) — UI
- [pandas](https://pandas.pydata.org/) — data handling
- [seaborn](https://seaborn.pydata.org/) + [matplotlib](https://matplotlib.org/) — charts
- `xlrd` — for reading legacy `.xls` files

## 🚀 Getting started

```bash
# 1. Clone the repo
git clone https://github.com/Shafiya0101/AI_Model_Comparison_Dashboard.git
cd AI_Model_Comparison_Dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run dashboard.py
```

Make sure your Excel result files are in the same folder as `dashboard.py`, then open the local URL Streamlit prints (usually http://localhost:8501).

## 📁 Project structure

```
AI_Model_Comparison_Dashboard/
├── dashboard.py          # Streamlit app
├── file1.xls             # Result set 1
├── file2.xlsx            # Result set 2
├── file3.xlsx            # Result set 3
└── requirements.txt      # Dependencies
```

## 💡 Possible improvements

- Add an answer-quality chart alongside the cost charts, so quality and cost sit side by side.
- Add a sidebar filter to compare a subset of models or prompt types.
- Add a screenshot of the dashboard to this README.

---

*A coursework project exploring sustainable / "green" AI — comparing models not just on answer quality but on the energy and carbon cost of each response.*
