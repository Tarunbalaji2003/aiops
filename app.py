import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Log Anomaly Detector", layout="wide")
st.title("🔍 Log Anomaly Detector")
st.write("Upload your `.txt` log file to detect anomalies using AI.")

uploaded_file = st.file_uploader("📁 Upload Log File", type="txt")

if uploaded_file is not None:
    try:
        # Attempt to decode log lines
        logs = uploaded_file.read().decode("utf-8", errors="ignore").splitlines()
        
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
        
        if not data:
            st.error("🚫 No valid log entries found. Ensure each log has timestamp, level, and message.")
        else:
            df = pd.DataFrame(data, columns=["timestamp", "level", "message"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["message_length"] = df["message"].apply(len)
            df["level_score"] = df["level"].map({
                "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4
            })

            # Filter out invalid log levels or timestamps
            df = df.dropna(subset=["timestamp", "level_score"])

            # Run anomaly detection
            model = IsolationForest(contamination=0.1, random_state=42)
            df["anomaly"] = model.fit_predict(df[["level_score", "message_length"]])
            df["is_anomaly"] = df["anomaly"].apply(lambda x: "❌ Anomaly" if x == -1 else "✅ Normal")

            st.success(f"✅ Processed {len(df)} logs. Detected {sum(df['is_anomaly'] == '❌ Anomaly')} anomalies.")

            st.subheader("📄 All Logs with Anomaly Tags")
            st.dataframe(df, use_container_width=True)

            st.subheader("🚨 Anomalies Only")
            anomalies = df[df["is_anomaly"] == "❌ Anomaly"]
            st.dataframe(anomalies, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error processing the file: {e}")
