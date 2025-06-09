import streamlit as st
import pandas as pd
import joblib
import os
import datetime

st.set_page_config(page_title="Prediction Form", page_icon="🔮", layout="wide")
st.title("🔮 Formulir Prediksi Permintaan")
st.markdown("Isi formulir di bawah ini untuk mendapatkan prediksi kategori permintaan.")

# Fungsi untuk memuat model dan objek lainnya
@st.cache_resource
def load_assets():
    if not os.path.exists('models/preprocessor.joblib'):
        return None, None, None, None, None
    
    preprocessor = joblib.load('models/preprocessor.joblib')
    le = joblib.load('models/label_encoder.joblib')
    feature_columns = joblib.load('models/feature_columns.joblib')
    
    models = {}
    model_files = {
        'K-Nearest Neighbors (KNN)': 'K-Nearest_Neighbors_(KNN).joblib',
        'Gaussian Naive Bayes': 'Gaussian_Naive_Bayes.joblib',
        'Nearest Centroid': 'Nearest_Centroid.joblib'
    }
    
    for name, file in model_files.items():
        try:
            models[name] = joblib.load(f'models/{file}')
        except FileNotFoundError:
            st.error(f"Model {name} tidak ditemukan. Silakan latih model terlebih dahulu di halaman 'Model Training'.")
            return None, None, None, None, None
            
    return preprocessor, le, models, feature_columns

assets = load_assets()
if assets is None or any(a is None for a in assets):
    st.warning("Aset model tidak lengkap. Harap jalankan pelatihan di halaman 'Model Training Results' terlebih dahulu.")
else:
    preprocessor, le, models, feature_columns = assets
    
    # Tampilkan Form
    with st.form("prediction_form"):
        st.header("Masukkan Fitur Produk")
        
        # Buat kolom agar rapi
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Menggunakan contoh dari data asli untuk opsi
            product_id = st.selectbox("ID Produk", options=['P0001', 'P0002', 'P0003', 'P0004', 'P0005', '...'])
            store_id = st.selectbox("ID Toko", options=['S0001', 'S0002', 'S0003', '...'])
            price = st.number_input("Harga (Price)", min_value=0.0, step=1.0, format="%.2f")
            
        with col2:
            discount = st.number_input("Diskon (Discount)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
            weather = st.selectbox("Kondisi Cuaca (Weather)", options=['Cloudy', 'Rainy', 'Sunny', 'Snowy'])
            promotion = st.selectbox("Promosi", options=['No', 'Yes'])
            
        with col3:
            date_input = st.date_input("Tanggal Prediksi", value=datetime.date.today())

        submit_button = st.form_submit_button(label="Lakukan Prediksi")

    if submit_button:
        # Buat dataframe dari input
        input_data = {
            'ProductID': [product_id],
            'StoreID': [store_id],
            'Price': [price],
            'Discount': [discount],
            'Weather': [weather],
            'Promotion': [promotion],
            'Month': [date_input.month],
            'DayOfWeek': [date_input.weekday()],
            'DayOfYear': [date_input.timetuple().tm_yday]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Pastikan kolom sesuai dengan saat training
        input_df = input_df[feature_columns]

        st.header("Hasil Prediksi Permintaan")
        
        col_res1, col_res2, col_res3 = st.columns(3)

        prediction_results = {}
        for name, model in models.items():
            # Prediksi
            prediction_encoded = model.predict(input_df)
            prediction_label = le.inverse_transform(prediction_encoded)
            prediction_results[name] = prediction_label[0]

        with col_res1:
            st.metric("Prediksi KNN", prediction_results.get('K-Nearest Neighbors (KNN)', 'Error'))
        with col_res2:
            st.metric("Prediksi Naive Bayes", prediction_results.get('Gaussian Naive Bayes', 'Error'))
        with col_res3:
            st.metric("Prediksi Nearest Centroid", prediction_results.get('Nearest Centroid', 'Error'))
