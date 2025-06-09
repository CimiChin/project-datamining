import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import os

st.set_page_config(page_title="Pelatihan Model", page_icon="🤖", layout="wide")
st.title("🤖 Pelatihan dan Evaluasi Model")

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    return pd.read_csv('retail_store_inventory.csv')

def train_and_evaluate():
    with st.spinner('Memuat data dan melakukan preprocessing...'):
        df = load_data()
        
        # 1. Membuat kategori target 'DemandCategory'
        bins = [0, 5, 10, float('inf')]
        labels = ['Rendah', 'Sedang', 'Tinggi']
        df['DemandCategory'] = pd.cut(df['SalesQuantity'], bins=bins, labels=labels, right=True)
        df.dropna(subset=['DemandCategory'], inplace=True)
        
        # 2. Feature Engineering dari Tanggal
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfYear'] = df['Date'].dt.dayofyear

        # 3. Label Encoding untuk target
        le = LabelEncoder()
        df['DemandCategoryEncoded'] = le.fit_transform(df['DemandCategory'])

        # 4. Mendefinisikan fitur (X) dan target (y)
        features_to_drop = ['Date', 'SalesQuantity', 'DemandCategory', 'DemandCategoryEncoded']
        X = df.drop(columns=features_to_drop)
        y = df['DemandCategoryEncoded']

        # 5. Preprocessing
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        # 6. Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    st.success("Preprocessing data selesai.")

    # 7. Definisikan Models
    models = {
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Gaussian Naive Bayes': GaussianNB(),
        'Nearest Centroid': NearestCentroid()
    }
    
    if not os.path.exists('models'):
        os.makedirs('models')

    # Simpan preprocessor, label encoder, dan kolom fitur
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')
    joblib.dump(list(X.columns), 'models/feature_columns.joblib')

    # Latih dan evaluasi setiap model
    for name, model in models.items():
        with st.spinner(f'Melatih model {name}...'):
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
            pipeline.fit(X_train, y_train)
            joblib.dump(pipeline, f'models/{name.replace(" ", "_")}.joblib')
            
            y_pred = pipeline.predict(X_test)
            
            st.subheader(f"Hasil untuk Model: {name}")
            accuracy = accuracy_score(y_test, y_pred)
            st.metric(label="Akurasi", value=f"{accuracy:.2%}")

            st.text("Laporan Klasifikasi:")
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            
            st.text("Confusion Matrix:")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, display_labels=le.classes_, cmap='Blues')
            st.pyplot(fig)
            st.markdown("---")
            
    st.success("Semua model telah berhasil dilatih dan dievaluasi!")

# Tombol untuk memulai pelatihan
if st.button("Mulai Proses Pelatihan Model"):
    train_and_evaluate()
else:
    st.info("Klik tombol di atas untuk memulai pelatihan dan evaluasi model.")
