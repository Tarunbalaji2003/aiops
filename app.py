import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

st.title("🔍 Log Anomaly Detector")
st.write("Upload your `system_logs.txt` file to detect anomalies.")

uploaded_file = st.file_uploader("Choose a log file", type="txt")

if uploaded_file is not None:
    logs = uploaded_file.readlines()
    logs = [line.decode("utf-8") for line in logs]

    # Parse logs
    data = []
    for log in logs:
        parts = log.strip().split(" ", 3)
        if len(parts) < 4:
            continue
        timestamp = parts[0] + " " + parts[1]
        level = parts[2]
        message = parts[3]
        data.append([timestamp, level, message])

    df = pd.DataFrame(data, columns=["timestamp", "level", "message"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["message_length"] = df["message"].apply(len)
    df["level_score"] = df["level"].map({"INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4})

    model = IsolationForest(contamination=0.1, random_state=42)
    df["anomaly"] = model.fit_predict(df[["level_score", "message_length"]])
    df["is_anomaly"] = df["anomaly"].apply(lambda x: "❌ Anomaly" if x == -1 else "✅ Normal")

    st.subheader("📄 All Logs with Anomaly Labels")
    st.dataframe(df)

    st.subheader("🚨 Detected Anomalies")
    st.dataframe(df[df["is_anomaly"] == "❌ Anomaly"])
