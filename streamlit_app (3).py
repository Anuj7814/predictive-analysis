
import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
import io
from datetime import datetime
import xlsxwriter

# Page config
st.set_page_config(page_title="Advanced Forecasting Tool", layout="wide")
st.title("📊 High-Accuracy Predictive Analysis (with Prophet)")

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
    component_images = {}
    accuracy_results = {}

    # Forecast logic with backtesting
    def forecast_metric(df, col_name):
        prophet_df = df[['Date', col_name]].dropna().rename(columns={'Date': 'ds', col_name: 'y'})

        # Handle zero or flatline data
        if prophet_df['y'].nunique() <= 1:
            return None, None, None, "⚠️ Not enough variation in data"

        model = Prophet()
        model.fit(prophet_df)

        # Forecast future
        future = model.make_future_dataframe(periods=1, freq='MS')
        forecast = model.predict(future)

        # Cross-validation
        try:
            df_cv = cross_validation(model, initial='365 days', period='30 days', horizon='30 days')
            df_p = performance_metrics(df_cv)
            mape = df_p['mape'].mean()
        except Exception as e:
            mape = None

        return model, forecast, mape, None

    # Forecast each metric
    for col in numeric_cols:
        if col in df.columns:
            model, forecast, mape, warning = forecast_metric(df, col)
            if warning:
                st.warning(f"{col}: {warning}")
                continue

            forecast_data[col] = forecast
            accuracy_results[col] = mape

            # Forecast plot
            fig = model.plot(forecast)
            plt.title(f"Forecast for {col}")
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            chart_images[col] = buf
            st.subheader(f"📊 Forecast: {col}")
            st.pyplot(fig)

            # Component plot
            comp_fig = model.plot_components(forecast)
            comp_buf = io.BytesIO()
            comp_fig.savefig(comp_buf, format="png")
            comp_buf.seek(0)
            component_images[col] = comp_buf
            st.subheader(f"🔍 Decomposition: {col}")
            st.pyplot(comp_fig)

            # Accuracy display
            if mape is not None:
                st.info(f"📉 MAPE for {col}: {round(mape * 100, 2)}%")

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
            worksheet.insert_image('A25', '', {'image_data': component_images[col], 'x_scale': 0.8, 'y_scale': 0.8})
            if accuracy_results.get(col) is not None:
                worksheet.write('A50', f"MAPE: {round(accuracy_results[col] * 100, 2)}%")

        writer.close()
        st.download_button(
            label="📥 Click to download Excel file",
            data=output.getvalue(),
            file_name=f"Monthly_Forecast_Report_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Please upload a CSV file to begin.")
