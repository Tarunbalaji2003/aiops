import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Log Anomaly Detector", layout="wide")

st.title("🔍 Log Anomaly Detector")
st.write("Upload a `.txt` log file with timestamps and levels to detect anomalies using Isolation Forest.")

uploaded_file = st.file_uploader("📁 Upload Log File", type=["txt"])

if uploaded_file:
    try:
        # Decode and read lines
        logs = uploaded_file.read().decode("utf-8").splitlines()

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

        # Build DataFrame
        df = pd.DataFrame(data, columns=["timestamp", "level", "message"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.dropna(subset=["timestamp"], inplace=True)
        df["message_length"] = df["message"].apply(len)
        df["level_score"] = df["level"].map({"INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}).fillna(0)

        # Show parsed logs
        st.subheader("📄 Parsed Logs Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Slider for contamination tuning
        contamination = st.slider("🔧 Set Anomaly Contamination Rate", 0.01, 0.5, 0.1, step=0.01)

        # Anomaly detection
        model = IsolationForest(contamination=contamination, random_state=42)
        df["anomaly"] = model.fit_predict(df[["level_score", "message_length"]])
        df["is_anomaly"] = df["anomaly"].apply(lambda x: "❌ Anomaly" if x == -1 else "✅ Normal")

        # Results
        st.subheader("📊 All Logs with Anomaly Labels")
        st.dataframe(df, use_container_width=True)

        st.subheader("🚨 Detected Anomalies")
        st.dataframe(df[df["is_anomaly"] == "❌ Anomaly"], use_container_width=True)

        # Optionally allow download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results as CSV", csv, "anomaly_results.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error processing the file: {e}")
