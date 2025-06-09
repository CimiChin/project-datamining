import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="EDA", page_icon="📈", layout="wide")

st.title("📈 Exploratory Data Analysis (EDA)")
st.markdown("Halaman ini menampilkan analisis data untuk memahami karakteristik dataset permintaan produk fashion.")

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    data = pd.read_csv('retail_store_inventory.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    return data

df = load_data()

# Tampilkan data mentah
st.header("1. Tampilan Awal Data")
st.dataframe(df.head())

# Tampilkan statistik deskriptif
st.header("2. Statistik Deskriptif")
st.write(df.describe())

# Visualisasi
st.header("3. Visualisasi Data")

col1, col2 = st.columns(2)

with col1:
    # Distribusi Kuantitas Penjualan
    st.subheader("Distribusi Kuantitas Penjualan (SalesQuantity)")
    fig_hist = px.histogram(df, x='SalesQuantity', nbins=50, title="Histogram Kuantitas Penjualan")
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    # Komposisi Cuaca
    st.subheader("Komposisi Kondisi Cuaca")
    weather_counts = df['Weather'].value_counts().reset_index()
    weather_counts.columns = ['Weather', 'Count']
    fig_pie_weather = px.pie(weather_counts, names='Weather', values='Count', title="Persentase Kondisi Cuaca")
    st.plotly_chart(fig_pie_weather, use_container_width=True)

# Tren Penjualan dari Waktu ke Waktu
st.subheader("Tren Total Penjualan Harian")
daily_sales = df.groupby('Date')['SalesQuantity'].sum().reset_index()
fig_line = px.line(daily_sales, x='Date', y='SalesQuantity', title='Total Penjualan Harian dari Waktu ke Waktu')
st.plotly_chart(fig_line, use_container_width=True)


# Hubungan antara Harga dan Kuantitas Penjualan
st.subheader("Hubungan antara Harga dan Kuantitas Penjualan")
fig_scatter = px.scatter(df, x='Price', y='SalesQuantity', color='Weather', 
                         title='Harga vs. Kuantitas Penjualan (diwarnai berdasarkan Cuaca)',
                         hover_data=['ProductID', 'Promotion'])
st.plotly_chart(fig_scatter, use_container_width=True)
