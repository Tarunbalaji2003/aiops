import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Log Anomaly Detector", layout="wide")
st.title("🔍 Log Anomaly Detector")
st.write("Upload a `.txt` log file with format: `<LEVEL> <MESSAGE>` (no timestamp).")

uploaded_file = st.file_uploader("Choose a log file", type="txt")

if uploaded_file is not None:
    try:
        # Read and decode the log file
        logs = uploaded_file.read().decode("utf-8", errors="ignore").splitlines()

        # Parse the logs
        data = []
        for log in logs:
            parts = log.strip().split(" ", 1)
            if len(parts) < 2:
                continue  # Skip malformed lines
            level = parts[0].upper()
            message = parts[1]
            data.append([level, message])

        if len(data) == 0:
            st.warning("⚠️ No valid log entries found. Please upload a properly formatted log file.")
        else:
            # Create DataFrame
            df = pd.DataFrame(data, columns=["level", "message"])
            df["message_length"] = df["message"].apply(len)

            # Map log levels to numeric values
            level_mapping = {"INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
            df["level_score"] = df["level"].map(level_mapping)

            # Drop rows with unknown levels
            df.dropna(subset=["level_score"], inplace=True)

            if df.empty:
                st.warning("⚠️ No valid rows after cleaning. Ensure levels are one of: INFO, WARNING, ERROR, CRITICAL.")
            else:
                # Apply Isolation Forest
                model = IsolationForest(contamination=0.1, random_state=42)
                df["anomaly"] = model.fit_predict(df[["level_score", "message_length"]])
                df["is_anomaly"] = df["anomaly"].apply(lambda x: "❌ Anomaly" if x == -1 else "✅ Normal")

                # Show all logs
                st.subheader("📄 All Logs with Anomaly Labels")
                st.dataframe(df)

                # Filter anomalies
                anomaly_df = df[df["is_anomaly"] == "❌ Anomaly"]

                st.subheader("🚨 Detected Anomalies")

                if not anomaly_df.empty:
                    filter_options = anomaly_df["level"].unique().tolist()
                    filter_options.sort()
                    selected_level = st.selectbox("🔎 Filter by Log Level", ["All"] + filter_options)

                    if selected_level != "All":
                        filtered_df = anomaly_df[anomaly_df["level"] == selected_level]
                    else:
                        filtered_df = anomaly_df

                    st.dataframe(filtered_df)
                else:
                    st.info("✅ No anomalies detected.")
    except Exception as e:
        st.error(f"❌ Error processing the file: {e}")
