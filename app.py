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

    ### 1. Analisis Data
    Halaman ini menampilkan analisis mendalam dari dataset. Anda dapat melihat statistik, distribusi data, dan visualisasi lainnya untuk memahami karakteristik data.

    ### 2. Pelatihan Model
    Di halaman ini, Anda akan memicu proses pelatihan model machine learning. Setelah model dilatih, halaman ini akan menampilkan hasil performanya.

    ### 3. Formulir Prediksi
    Gunakan halaman ini untuk melakukan prediksi permintaan secara real-time. Cukup isi formulir dengan fitur-fitur yang relevan, dan model akan memberikan prediksi kategori permintaan (Rendah, Sedang, atau Tinggi).
    
    ---
    
    **Petunjuk Penggunaan:**
    1.  Buka halaman **Pelatihan Model**.
    2.  Klik tombol "Mulai Proses Pelatihan Model" untuk melatih dan menyimpan model.
    3.  Setelah selesai, Anda bisa menjelajahi semua halaman.
    
    """
)
