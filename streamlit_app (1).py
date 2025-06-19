
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import io
from datetime import datetime
import xlsxwriter

# Page config
st.set_page_config(page_title="Monthly Forecast Tool", layout="wide")
st.title("📈 Predictive Analysis for Monthly Brand Metrics")

# Upload CSV
uploaded_file = st.file_uploader("Upload your monthly CSV file", type=["csv"])

if uploaded_file:
    # Read and clean data
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    if 'Month' not in df.columns:
        df.columns.values[0] = 'Month'

    df['Date'] = pd.to_datetime(df['Month'], format='%Y-%m', errors='coerce')

    numeric_cols = ['Total Orders', 'Overall Customers', 'Overall Sales', 'Business ATV']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True).str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    st.success("File uploaded and cleaned successfully!")
    st.dataframe(df[['Date'] + numeric_cols])

    forecast_data = {}
    chart_images = {}

    # Forecast logic
    def forecast_metric(df, col_name):
        prophet_df = df[['Date', col_name]].dropna().rename(columns={'Date': 'ds', col_name: 'y'})
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=1, freq='MS')
        forecast = model.predict(future)
        return forecast

    # Forecast each metric
    for col in numeric_cols:
        if col in df.columns:
            forecast = forecast_metric(df, col)
            forecast_data[col] = forecast

            # Plot
            fig = Prophet().plot(forecast)
            plt.title(f"Forecast for {col}")
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            chart_images[col] = buf
            st.subheader(f"📊 Forecast: {col}")
            st.pyplot(fig)

    # Download forecast as Excel
    if st.button("📥 Download Forecast Report (Excel)"):
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')

        for col, forecast in forecast_data.items():
            forecast_out = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(1)
            forecast_out.insert(0, 'Metric', col)
            forecast_out.to_excel(writer, sheet_name=col[:31], index=False, startrow=0)
            worksheet = writer.sheets[col[:31]]
            worksheet.insert_image('A8', '', {'image_data': chart_images[col], 'x_scale': 0.8, 'y_scale': 0.8})

        writer.close()
        st.download_button(
            label="📥 Click to download Excel file",
            data=output.getvalue(),
            file_name=f"Monthly_Forecast_Report_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Please upload a CSV file to begin.")
