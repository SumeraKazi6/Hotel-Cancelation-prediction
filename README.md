# Hotel Booking Cancellation Prediction — Streamlit deployment

This web application uses the **same trained preprocessing + classifier pipeline** saved by the ML mini-project notebook. The notebook's saved run reports the selected model as **Random Forest trained on Original Training Data**. The app does not retrain or substitute a different model.

## 1. Create the model file

The notebook saves `hotel_cancellation_model.pkl` in its runtime when the model-training and saving cells are executed.

In Google Colab, run the notebook through the model-saving cell (the cell containing `joblib.dump(best_pipeline, model_filename)`). Then run:

```python
from google.colab import files
files.download("hotel_cancellation_model.pkl")
```

Download the file and place it beside `app.py` in this folder. The pickle file is intentionally not included in this ZIP because it must be generated from your actual executed notebook and dataset.

## 2. Run locally

Install Python 3.10 or 3.11, open a terminal in this folder and run:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The browser will open the local application.

## 3. Deploy on Streamlit Community Cloud

1. Create a GitHub repository, for example `hotel-cancellation-prediction`.
2. Upload `app.py`, `requirements.txt`, `README.md`, and your generated `hotel_cancellation_model.pkl`.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
5. Deploy. Streamlit will install the requirements and publish a web URL.

Keep the repository public/private according to your course requirements. Do not commit personal data or credentials.

## Feature compatibility

The input fields and engineered features follow the notebook's `X` feature schema (31 predictors). The app derives `total_stay`, `total_guests`, and `total_previous_bookings` before prediction. `reservation_status`, `reservation_status_date`, `company`, and the target `is_canceled` are not sent to the model.

## Important

Only load a model file that you created and trust. `joblib`/pickle files can execute code when loaded. The model's predictions are estimates for an academic demonstration and should not be used as the sole basis for consequential decisions.
