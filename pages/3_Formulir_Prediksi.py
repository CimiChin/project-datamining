import streamlit as st
import pandas as pd
import joblib
import os
import datetime

st.set_page_config(page_title="Formulir Prediksi", page_icon="🔮", layout="wide")
st.title("🔮 Formulir Prediksi Permintaan")

# Fungsi untuk memuat aset model
@st.cache_resource
def load_assets():
    models_path = 'models'
    if not os.path.exists(models_path) or not os.listdir(models_path):
        return None
    
    assets = {
        'preprocessor': joblib.load(os.path.join(models_path, 'preprocessor.joblib')),
        'label_encoder': joblib.load(os.path.join(models_path, 'label_encoder.joblib')),
        'feature_columns': joblib.load(os.path.join(models_path, 'feature_columns.joblib')),
        'models': {
            'K-Nearest Neighbors': joblib.load(os.path.join(models_path, 'K-Nearest_Neighbors.joblib')),
            'Gaussian Naive Bayes': joblib.load(os.path.join(models_path, 'Gaussian_Naive_Bayes.joblib')),
            'Nearest Centroid': joblib.load(os.path.join(models_path, 'Nearest_Centroid.joblib'))
        }
    }
    return assets

assets = load_assets()

if assets is None:
    st.warning("Model belum dilatih. Silakan pergi ke halaman 'Pelatihan Model' dan klik tombol untuk melatih model terlebih dahulu.")
else:
    st.markdown("Isi formulir di bawah ini untuk mendapatkan prediksi kategori permintaan.")
    
    with st.form("prediction_form"):
        st.header("Masukkan Fitur Produk")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            product_id = st.selectbox("ID Produk", options=['P0001', 'P0002', 'P0003', 'P0004', 'P0005', 'P0006', 'P0007', 'P0008', 'P0009', 'P0010'])
            store_id = st.selectbox("ID Toko", options=['S0001', 'S0002', 'S0003', 'S0004'])
            price = st.number_input("Harga (Price)", min_value=0.0, step=1.0, value=50.0, format="%.2f")
            
        with col2:
            discount = st.number_input("Diskon (Discount)", min_value=0.0, max_value=1.0, step=0.01, value=0.1, format="%.2f")
            weather = st.selectbox("Kondisi Cuaca (Weather)", options=['Cloudy', 'Rainy', 'Sunny', 'Snowy'])
            promotion = st.selectbox("Promosi", options=['No', 'Yes'])
            
        with col3:
            date_input = st.date_input("Tanggal Prediksi", value=datetime.date.today())

        submit_button = st.form_submit_button(label="Lakukan Prediksi")

    if submit_button:
        # Buat dataframe dari input
        input_data = {
            'ProductID': product_id,
            'StoreID': store_id,
            'Price': price,
            'Discount': discount,
            'Weather': weather,
            'Promotion': promotion,
            'Month': date_input.month,
            'DayOfWeek': date_input.weekday(),
            'DayOfYear': date_input.timetuple().tm_yday
        }
        
        input_df = pd.DataFrame([input_data])
        # Pastikan urutan kolom sesuai dengan saat pelatihan
        input_df = input_df[assets['feature_columns']]

        st.header("Hasil Prediksi Permintaan")
        
        col_res1, col_res2, col_res3 = st.columns(3)

        # Lakukan prediksi untuk setiap model
        prediction_label_knn = assets['label_encoder'].inverse_transform(assets['models']['K-Nearest Neighbors'].predict(input_df))[0]
        prediction_label_nb = assets['label_encoder'].inverse_transform(assets['models']['Gaussian Naive Bayes'].predict(input_df))[0]
        prediction_label_nc = assets['label_encoder'].inverse_transform(assets['models']['Nearest Centroid'].predict(input_df))[0]
        
        with col_res1:
            st.metric("Prediksi KNN", prediction_label_knn)
        with col_res2:
            st.metric("Prediksi Naive Bayes", prediction_label_nb)
        with col_res3:
            st.metric("Prediksi Nearest Centroid", prediction_label_nc)
