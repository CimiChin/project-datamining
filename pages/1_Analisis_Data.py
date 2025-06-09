import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Analisis Data", page_icon="📈", layout="wide")
st.title("📈 Analisis Data Eksploratif (EDA)")
st.markdown("Halaman ini menampilkan analisis untuk memahami karakteristik dataset.")

# Fungsi untuk memuat data dengan caching agar lebih cepat
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('retail_store_inventory.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    except FileNotFoundError:
        st.error("File 'retail_store_inventory.csv' tidak ditemukan. Pastikan file tersebut ada di folder utama proyek.")
        return None

df = load_data()

if df is not None:
    st.header("1. Tampilan Awal Data")
    st.dataframe(df.head())

    st.header("2. Statistik Deskriptif")
    st.write(df.describe())

    st.header("3. Visualisasi Data")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Kuantitas Penjualan")
        # Add these lines before the px.histogram call
        print(df.info())
        print(df['SalesQuantity'].head())
        print(df['SalesQuantity'].describe())
        fig_hist = px.histogram(df, x='SalesQuantity', nbins=50, title="Histogram Kuantitas Penjualan")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("Komposisi Kondisi Cuaca")
        weather_counts = df['Weather'].value_counts().reset_index()
        weather_counts.columns = ['Weather', 'Count']
        fig_pie = px.pie(weather_counts, names='Weather', values='Count', title="Persentase Kondisi Cuaca")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Tren Total Penjualan Harian")
    daily_sales = df.groupby('Date')['SalesQuantity'].sum().reset_index()
    fig_line = px.line(daily_sales, x='Date', y='SalesQuantity', title='Total Penjualan Harian')
    st.plotly_chart(fig_line, use_container_width=True)
