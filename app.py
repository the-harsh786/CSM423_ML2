import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="LoanLens | Repayment Insights",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f5f7fb; }
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1400px; }
.hero { padding: 1.7rem 1.9rem; border-radius: 22px; color: white;
        background: linear-gradient(120deg,#111827 0%,#172554 55%,#2563eb 100%);
        margin-bottom: 1.2rem; }
.hero h1 { font-family: 'Manrope', sans-serif; font-size: 2.25rem; margin: 0 0 .35rem 0; }
.hero p { color: #dbeafe; font-size: 1rem; margin: 0; }
.kpi { background: white; border: 1px solid #e5eaf3; border-radius: 16px; padding: 1rem 1.15rem; }
.kpi-label { color:#64748b; font-size:.85rem; margin-bottom:.35rem; }
.kpi-value { color:#0f172a; font-family:'Manrope',sans-serif; font-size:1.55rem; font-weight:800; }
.section-title { color:#0f172a; font-family:'Manrope',sans-serif; font-weight:800; font-size:1.15rem; margin: .6rem 0 .3rem; }
.small-muted { color:#64748b; font-size:.88rem; }
div[data-testid="stForm"] { background:white; border:1px solid #e5eaf3; padding:1.2rem; border-radius:18px; }
.stButton>button, div[data-testid="stFormSubmitButton"] button { border-radius:10px; font-weight:700; min-height:2.8rem; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "loan_repayment_model.pkl"
DATA_PATH = "original_dataset.csv"
TARGET = "loan_paid_back"

with st.sidebar:
    st.markdown("## 💳 LoanLens")
    st.caption("Loan repayment prediction")
    st.divider()
    page = st.radio("NAVIGATION", ["Predict repayment", "Dataset explorer", "About the model"], index=0)
    st.divider()
    st.markdown("**Project**")
    st.caption("Machine Learning · Classification")
    st.caption("Models: Logistic Regression & Linear SVM")
    st.caption("Built with Streamlit + scikit-learn")

st.markdown("""
<div class="hero">
  <h1>LoanLens</h1>
  <p>Loan repayment insights, powered by machine learning.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)

bundle = load_bundle()
df = load_data()

if bundle is None:
    st.error("Model file not found: `loan_repayment_model.pkl`.")
    st.markdown("Run the notebook's final training/save cells, then place the exported model in this app folder.")
    st.stop()
if df is None:
    st.error("Dataset not found: `original_dataset.csv`.")
    st.markdown("Place the same CSV used in your notebook beside `app.py`. It is used to create sensible input fields and power the Dataset Explorer.")
    st.stop()

model = bundle["model"]
features = bundle.get("features", [c for c in df.columns if c != TARGET])
feature_df = df[features].copy()
numeric_cols = feature_df.select_dtypes(include="number").columns.tolist()
categorical_cols = [c for c in features if c not in numeric_cols]

if page == "Predict repayment":
    st.markdown('<div class="section-title">Applicant profile</div>', unsafe_allow_html=True)
    st.markdown('<p class="small-muted">Enter the applicant details, then run the model. This is an educational prediction—not a lending decision.</p>', unsafe_allow_html=True)

    with st.form("applicant_form"):
        st.markdown("#### Personal & employment details")
        personal = [c for c in ["age", "gender", "marital_status", "education_level", "employment_status"] if c in features]
        cols = st.columns(3)
        values = {}
        for i, col in enumerate(personal):
            s = feature_df[col]
            with cols[i % 3]:
                if col in numeric_cols:
                    values[col] = st.number_input(col.replace("_", " ").title(), min_value=float(s.min()), max_value=float(s.max()), value=float(s.median()), step=1.0)
                else:
                    opts = sorted(s.dropna().astype(str).unique().tolist())
                    values[col] = st.selectbox(col.replace("_", " ").title(), opts)

        st.markdown("#### Financial & credit details")
        remaining = [c for c in features if c not in personal]
        cols = st.columns(3)
        for i, col in enumerate(remaining):
            s = feature_df[col]
            with cols[i % 3]:
                label = col.replace("_", " ").title()
                if col in numeric_cols:
                    lo, hi = float(s.min()), float(s.max())
                    med = float(s.median())
                    # Keep a valid range even for constant columns.
                    if lo == hi:
                        values[col] = st.number_input(label, value=lo, disabled=True)
                    elif pd.api.types.is_integer_dtype(s.dtype):
                        values[col] = st.number_input(label, min_value=int(lo), max_value=int(hi), value=int(round(med)), step=1)
                    else:
                        values[col] = st.number_input(label, min_value=lo, max_value=hi, value=med, step=max((hi-lo)/100, 0.01), format="%.3f" if "ratio" in col else "%.2f")
                else:
                    opts = sorted(s.dropna().astype(str).unique().tolist())
                    values[col] = st.selectbox(label, opts if opts else ["Unknown"])

        submitted = st.form_submit_button("✨ Predict repayment outcome", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{c: values[c] for c in features}], columns=features)
        try:
            pred = int(model.predict(row)[0])
            st.divider()
            st.markdown("### Prediction result")
            left, right = st.columns([1, 2])
            with left:
                if pred == 1:
                    st.success("Likely to repay")
                    st.markdown("## ✅ Loan repaid")
                else:
                    st.warning("Higher repayment risk")
                    st.markdown("## ⚠️ Loan not repaid")
            with right:
                st.markdown("**What this means**")
                if hasattr(model, "decision_function"):
                    st.write("The model classified this applicant using the patterns learned from the training dataset.")
                else:
                    st.write("The model returned a class based on the training data patterns.")
                st.caption("The output is a model prediction, not a probability or a guarantee. Do not use it as the sole basis for a real credit decision.")
                st.download_button("Download prediction as CSV", row.assign(predicted_loan_paid_back=pred).to_csv(index=False).encode("utf-8"), "loan_prediction.csv", "text/csv")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.caption("Check that the model was trained using the same feature columns as this dataset.")

elif page == "Dataset explorer":
    st.markdown('<div class="section-title">Dataset overview</div>', unsafe_allow_html=True)
    a,b,c,d = st.columns(4)
    with a: st.markdown(f'<div class="kpi"><div class="kpi-label">Records</div><div class="kpi-value">{len(df):,}</div></div>', unsafe_allow_html=True)
    with b: st.markdown(f'<div class="kpi"><div class="kpi-label">Input features</div><div class="kpi-value">{len(features)}</div></div>', unsafe_allow_html=True)
    with c: st.markdown(f'<div class="kpi"><div class="kpi-label">Missing cells</div><div class="kpi-value">{int(df.isna().sum().sum()):,}</div></div>', unsafe_allow_html=True)
    with d:
        rate = df[TARGET].mean()*100 if TARGET in df else 0
        st.markdown(f'<div class="kpi"><div class="kpi-label">Repaid in dataset</div><div class="kpi-value">{rate:.1f}%</div></div>', unsafe_allow_html=True)
    st.write("")
    tab1, tab2 = st.tabs(["Preview data", "Target distribution"])
    with tab1:
        st.dataframe(df.head(100), use_container_width=True, hide_index=True)
        st.download_button("Download dataset CSV", df.to_csv(index=False).encode("utf-8"), "loan_dataset.csv", "text/csv")
    with tab2:
        if TARGET in df:
            counts = df[TARGET].value_counts().sort_index()
            chart = pd.DataFrame({"Outcome": [("Not repaid (0)" if str(k) == "0" else "Repaid (1)") for k in counts.index], "Applicants": counts.values}).set_index("Outcome")
            st.bar_chart(chart)
            st.caption("Target label: 0 = not repaid, 1 = repaid.")

else:
    st.markdown('<div class="section-title">About this project</div>', unsafe_allow_html=True)
    st.write("LoanLens is a student machine-learning project that predicts whether a loan is repaid (1) or not repaid (0), using applicant, income, loan, and credit-related features.")
    st.markdown("#### Model workflow")
    st.markdown("""
    1. Split the data into training and test sets using stratification.
    2. Fill missing numerical values with the median and scale numerical features.
    3. Fill missing categorical values with the most frequent category and one-hot encode them.
    4. Train Logistic Regression and a Linear Support Vector Machine (SVM).
    5. Tune the SVM regularization parameter with 5-fold stratified cross-validation.
    6. Evaluate with accuracy, macro F1-score, classification report, and a confusion matrix.
    """)
    st.markdown("#### Notebook results")
    m1,m2,m3 = st.columns(3)
    with m1: st.metric("Logistic Regression accuracy", "88.68%")
    with m2: st.metric("Linear SVM accuracy", "88.78%")
    with m3: st.metric("SVM macro F1", "0.800")
    st.info("These are the results recorded in the supplied notebook. Results can differ if the dataset or training setup changes.")
    st.warning("Fairness & limitations: the dataset may contain sensitive or proxy attributes. This model can make mistakes and should not be used to approve or reject real loan applications.")
