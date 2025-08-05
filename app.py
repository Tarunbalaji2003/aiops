import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Log Anomaly Detector", layout="wide")
st.title("🔍 Log Anomaly Detector")
st.write("Upload a `.txt` log file with timestamps to detect anomalies using AI.")

# Upload log file
uploaded_file = st.file_uploader("📂 Choose a log file", type="txt")

if uploaded_file is not None:
    try:
        # Decode and read file lines
        logs = uploaded_file.read().decode("utf-8", errors="ignore").splitlines()
        
        # Parse logs into structured format
        data = []
        for log in logs:
            parts = log.strip().split(" ", 3)
            if len(parts) < 4:
                continue  # skip malformed lines
            timestamp = parts[0] + " " + parts[1]
            level = parts[2]
            message = parts[3]
            data.append([timestamp, level, message])

        # Create DataFrame
        df = pd.DataFrame(data, columns=["timestamp", "level", "message"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
        df["message_length"] = df["message"].apply(len)
        df["level_score"] = df["level"].map({
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4
        })

        # Drop rows with invalid timestamps or unknown levels
        df = df.dropna(subset=["timestamp", "level_score"])

        if df.empty:
            st.warning("⚠️ No valid log entries found. Please upload a properly formatted log file.")
        else:
            # Show parsed data preview
            st.subheader("👀 Parsed Logs Preview")
            st.dataframe(df.head(), use_container_width=True)

            # Fit Isolation Forest model
            model = IsolationForest(contamination=0.1, random_state=42)
            df["anomaly"] = model.fit_predict(df[["level_score", "message_length"]])
            df["is_anomaly"] = df["anomaly"].apply(lambda x: "❌ Anomaly" if x == -1 else "✅ Normal")

            st.success(f"✅ Processed {len(df)} logs. Detected {sum(df['is_anomaly'] == '❌ Anomaly')} anomalies.")

            # Show all logs
            st.subheader("📄 All Logs with Anomaly Status")
            st.dataframe(df, use_container_width=True)

            # Show only anomalies
            st.subheader("🚨 Detected Anomalies")
            anomalies = df[df["is_anomaly"] == "❌ Anomaly"]
            if anomalies.empty:
                st.info("✅ No anomalies detected in the logs.")
            else:
                st.dataframe(anomalies, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error processing the file: {e}")
