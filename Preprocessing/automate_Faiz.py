import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Fungsi untuk memuat dataset raw."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Fungsi otomatisasi transformasi data."""
    # 1. Menangani Missing Values
    df['assist_player'] = df['assist_player'].fillna('No Assist')
    df.dropna(subset=['goal_method'], inplace=True) 
    df = df[df['goal_method'] != 'Not reported']

    # 2. Seleksi Fitur
    kolom_dihapus = ['match_date', 'final_score', 'score_after_goal']
    df.drop(columns=[col for col in kolom_dihapus if col in df.columns], inplace=True)

    # 3. Memisahkan Target dan Fitur
    X = df.drop(columns=['goal_method'])
    y = df['goal_method']

    # 4. Encoding Target (y)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    df_y = pd.DataFrame(y_encoded, columns=['goal_method_encoded'])

    # 5. One-Hot Encoding Fitur Kategorikal (X)
    kolom_kategorikal = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=kolom_kategorikal, drop_first=True)

    # 6. Standarisasi Fitur (X)
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X_encoded)
    df_X = pd.DataFrame(X_scaled_array, columns=X_encoded.columns)

    # 7. Menggabungkan X dan y kembali menjadi satu dataset yang siap dilatih
    df_final = pd.concat([df_X, df_y.reset_index(drop=True)], axis=1)
    
    return df_final

if __name__ == "__main__":
    # Mengatur path secara dinamis agar bisa dijalankan di lingkungan mana saja (lokal maupun GitHub Actions)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_PATH = os.path.join(BASE_DIR, 'ronaldo_goals_kaggle_ready.csv')
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'Preprocessing', 'ronaldo_preprocessed.csv')

    print("=== Memulai Pipeline Preprocessing ===")
    
    # Proses Load
    print(f"[1/3] Membaca data mentah dari: {RAW_DATA_PATH}")
    raw_df = load_data(RAW_DATA_PATH)
    
    # Proses Preprocessing
    print("[2/3] Memproses data (Pembersihan, Encoding, & Standarisasi)...")
    processed_df = preprocess_data(raw_df)
    
    # Proses Save
    print(f"[3/3] Menyimpan data yang siap dilatih ke: {PROCESSED_DATA_PATH}")
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    
    print("=== Pipeline Selesai! Data siap digunakan untuk Kriteria 2 ===")