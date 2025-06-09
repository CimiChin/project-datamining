import streamlit as st

st.set_page_config(
    page_title="Prediksi Permintaan Fashion",
    page_icon="👕",
    layout="wide"
)

st.title("👕 Dashboard Prediksi Permintaan Produk Fashion")

st.sidebar.success("Pilih halaman di atas untuk memulai.")

st.markdown(
    """
    Selamat datang di Dashboard Prediksi Permintaan Produk Fashion.
    
    Dashboard ini terdiri dari tiga halaman utama yang dapat Anda akses melalui menu di sebelah kiri:

    ### 1. 📈 Exploratory Data Analysis (EDA)
    Halaman ini menampilkan analisis mendalam dari dataset. Anda dapat melihat statistik deskriptif, distribusi data, dan visualisasi lainnya untuk memahami karakteristik data permintaan produk fashion.

    ### 2. 🤖 Model Training Results
    Di halaman ini, Anda akan melihat hasil dari pelatihan model machine learning. Kami mengevaluasi performa model menggunakan metrik seperti akurasi, presisi, dan recall, serta menampilkan confusion matrix.

    ### 3. 🔮 Prediction Form
    Gunakan halaman ini untuk melakukan prediksi permintaan secara real-time. Cukup isi formulir dengan fitur-fitur yang relevan, dan model akan memberikan prediksi kategori permintaan (Rendah, Sedang, atau Tinggi).
    
    **Dataset yang digunakan**: `retail_store_inventory.csv`
    
    Silakan mulai dengan menjelajahi halaman **EDA** untuk mendapatkan wawasan tentang data!
    """
)
