# LoanLens — Loan Repayment Prediction App

A Streamlit front end for the ML2Project notebook. The app uses the saved scikit-learn pipeline to classify loan repayment as `1 = repaid` or `0 = not repaid`.

## Files you must add
Place these two files in the same folder as `app.py`:
- `original_dataset.csv` — the same dataset used in your notebook (20,000 rows, 22 columns according to the notebook output).
- `loan_repayment_model.pkl` — exported model bundle from the notebook's save-model cell.

The `.pkl` file is not included because it was not attached with the notebook.

## Export the model from your notebook
Run the cells that train and evaluate the SVM, then run the final “Save your trained model” cell. It should create `loan_repayment_model.pkl`. If running locally, make sure `original_dataset.csv` is in the notebook's working directory.

Your notebook's save cell should save a dictionary like:
```python
joblib.dump(
    {"model": best_model, "features": X.columns.tolist()},
    "loan_repayment_model.pkl"
)
```

## Run locally on Windows
Open PowerShell in this folder and run:
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal and activate again.

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository, for example `loanlens-streamlit`.
2. Upload `app.py`, `requirements.txt`, `original_dataset.csv`, and `loan_repayment_model.pkl`.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
5. Deploy and wait for the build to finish. Copy the public app URL.

**Privacy:** only upload a dataset you are permitted to publish. A public GitHub repository and deployed app may expose files committed to it. If the dataset is private or too large, store it outside the repository and configure a secure download process instead.

## Important
This is an educational model, not a financial decision system. It provides a class prediction, not a calibrated probability. Do not use it to make real lending decisions.
