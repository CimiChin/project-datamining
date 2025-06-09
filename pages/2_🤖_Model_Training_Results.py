import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import joblib
import os

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")
st.title("🤖 Hasil Pelatihan Model")

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    data = pd.read_csv('retail_store_inventory.csv')
    return data

# Fungsi untuk mengubah SalesQuantity menjadi kategori
def create_demand_category(df):
    bins = [0, 5, 10, float('inf')]
    labels = ['Rendah', 'Sedang', 'Tinggi']
    df['DemandCategory'] = pd.cut(df['SalesQuantity'], bins=bins, labels=labels, right=True)
    df.dropna(subset=['DemandCategory'], inplace=True)
    return df

# Fungsi untuk melatih dan menyimpan model
def train_and_save_models(df):
    st.write("Memulai proses preprocessing dan pelatihan...")
    
    # 1. Feature Engineering dari tanggal
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear

    # 2. Membuat kategori target
    df = create_demand_category(df)
    
    # 3. Label Encoding untuk target
    le = LabelEncoder()
    df['DemandCategoryEncoded'] = le.fit_transform(df['DemandCategory'])

    # 4. Mendefinisikan fitur dan target
    X = df.drop(['Date', 'SalesQuantity', 'DemandCategory', 'DemandCategoryEncoded'], axis=1)
    y = df['DemandCategoryEncoded']
    
    # 5. Preprocessing
    numeric_features = ['Price', 'Discount', 'Month', 'DayOfWeek', 'DayOfYear']
    categorical_features = ['ProductID', 'StoreID', 'Weather', 'Promotion']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # 6. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 7. Definisikan Models
    models = {
        'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5),
        'Gaussian Naive Bayes': GaussianNB(),
        'Nearest Centroid': NearestCentroid()
    }
    
    results = {}

    if not os.path.exists('models'):
        os.makedirs('models')

    # Simpan preprocessor dan label encoder
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')
    joblib.dump((X_test, y_test), 'models/test_data.joblib')
    joblib.dump(list(X.columns), 'models/feature_columns.joblib')

    for name, model in models.items():
        st.write(f"Melatih model: {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, f'models/{name.replace(" ", "_")}.joblib')
        st.write(f"Model {name} berhasil dilatih dan disimpan.")
    
    return True


# Tombol untuk memulai pelatihan
if st.button("Mulai Pelatihan dan Evaluasi Model"):
    with st.spinner('Sedang melatih model, ini mungkin memakan waktu beberapa saat...'):
        df = load_data()
        train_and_save_models(df)
    st.success("Semua model telah berhasil dilatih dan disimpan di folder `models/`!")
    st.info("Hasil evaluasi model sekarang tersedia di bawah.")

# Tampilkan hasil jika model sudah ada
if os.path.exists('models/preprocessor.joblib'):
    st.header("Evaluasi Kinerja Model")
    
    # Muat data uji, preprocessor, dan label encoder
    X_test, y_test = joblib.load('models/test_data.joblib')
    le = joblib.load('models/label_encoder.joblib')
    
    models_list = ['K-Nearest_Neighbors_(KNN).joblib', 'Gaussian_Naive_Bayes.joblib', 'Nearest_Centroid.joblib']
    
    for model_file in models_list:
        model_name = model_file.replace('_', ' ').replace('.joblib', '')
        st.subheader(f"Hasil untuk Model: {model_name}")
        
        pipeline = joblib.load(f'models/{model_file}')
        y_pred = pipeline.predict(X_test)
        
        # Tampilkan metrik
        accuracy = accuracy_score(y_test, y_pred)
        st.metric(label="Akurasi", value=f"{accuracy:.2%}")

        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        st.text("Laporan Klasifikasi:")
        st.dataframe(pd.DataFrame(report).transpose())
        
        # Tampilkan Confusion Matrix
        st.text("Confusion Matrix:")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, display_labels=le.classes_, cmap='Blues')
        st.pyplot(fig)
        st.markdown("---")
else:
    st.warning("Model belum dilatih. Klik tombol di atas untuk memulai proses pelatihan.")
