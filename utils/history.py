import os
import pandas as pd
from datetime import datetime

HISTORY_FILE = "prediction_history.csv"

def save_prediction(disease, confidence):
    new_record = pd.DataFrame({
        "Date": [datetime.now().strftime("%d-%m-%Y")],
        "Time": [datetime.now().strftime("%H:%M:%S")],
        "Disease": [disease],
        "Confidence": [round(confidence, 2)]
    })

    if os.path.exists(HISTORY_FILE):
        new_record.to_csv(
            HISTORY_FILE,
            mode="a",
            header=False,
            index=False
        )
    else:
        new_record.to_csv(
            HISTORY_FILE,
            index=False
        )

def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["Date", "Time", "Disease", "Confidence"])