# Dashboard Prediksi Permintaan Produk Fashion

Ini adalah proyek Streamlit untuk membuat dasbor interaktif yang memprediksi tingkat permintaan produk fashion berdasarkan data historis.

## Tinjauan Proyek

Dasbor ini terdiri dari 3 halaman utama:
1.  **📈 Exploratory Data Analysis (EDA)**: Menganalisis dan memvisualisasikan dataset untuk menemukan wawasan.
2.  **🤖 Model Training Results**: Melatih model machine learning (KNN, Naive Bayes, Nearest Centroid) dan menampilkan hasil evaluasinya.
3.  **🔮 Prediction Form**: Formulir interaktif untuk melakukan prediksi permintaan baru berdasarkan input pengguna.

## Cara Menjalankan Proyek

1.  **Clone Repositori**
    ```bash
    git clone [URL-repositori-Anda]
    cd fashion-demand-prediction
    ```

2.  **Buat Lingkungan Virtual (Opsional tapi Direkomendasikan)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Di Windows: venv\Scripts\activate
    ```

3.  **Install Dependensi**
    Pastikan Anda memiliki file `retail_store_inventory.csv` di folder utama proyek.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan Aplikasi Streamlit**
    ```bash
    streamlit run app.py
    ```

5.  **Gunakan Aplikasi**
    - Buka browser Anda dan akses URL lokal (biasanya `http://localhost:8501`).
    - Pertama, buka halaman **Model Training Results** dan klik tombol "Mulai Pelatihan" untuk membuat file model.
    - Setelah pelatihan selesai, Anda bisa menjelajahi semua halaman, termasuk melakukan prediksi di **Prediction Form**.
