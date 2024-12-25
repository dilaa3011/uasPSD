import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from component.nav import navbar
from component.bootstrap import bootstrap


# ------------------navbar-----------------------
navbar()
def deskPre() :
    bootstrap()
    desk_pre = """
    <p>
        Preprocesing data adalah proses untuk mengubah data mentah menjadi lebih teratur agar ketika dilakukan teknik data mining tingkat akurasinya lebih tinggi.
    </p>
    <br>
    <div class="list-group">
        <ol class="list-group list-group-numbered">
            <li class="list-group-item">
            <strong>Pembersihan Data (Data Cleaning):</strong>
            Data cleaning atau pembersihan data terutama dilakukan sebagai bagian dari data preprocessing untuk membersihkan data dengan mengisi nilai yang hilang, menghaluskan data yang noise, menyelesaikan data yang tidak konsisten, dan menghapus outlier.
            </li>
            <li class="list-group-item">
            <strong>Penggabungan Data (Data Integration):</strong>
            Integrasi data adalah salah satu langkah data preprocessing yang digunakan untuk menggabungkan data yang ada di berbagai sumber menjadi satu penyimpanan data yang lebih besar seperti gudang data atau data warehouse.</li>
            <li class="list-group-item">
            <strong>Synthetic Minority Teknik Oversampling (SMOTE):</strong>
            Metode SMOTE digunakan untuk menyeimbangkan data yang minoritas sehingga distribusi setiap kelasnya seimbang dan dapat mengingkatkan performa model.</li>
        </ol>
    </div>
    """
    st.markdown(desk_pre,unsafe_allow_html=True)
def pre():
    bootstrap()
    # ------------------FULL DATA-----------------------
    data = pd.read_excel('wine_quality.xlsx')
 
    X = data.drop('color', axis=1)
    y = data['color']

    # Normalisasi data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # SMOTE untuk mengatasi data tidak seimbang
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    X_test_smote, y_test_smote = smote.fit_resample(X_test, y_test)

    # Tampilkan hasil ke halaman Streamlit
    st.write("### Data Train Sebelum SMOTE")
    st.write(y_train.value_counts())
    st.write("### Data Train Setelah SMOTE")
    st.write(y_train_smote.value_counts())
    st.write("### Data Test Sebelum SMOTE")
    st.write(y_test.value_counts())
    st.write("### Data Test Setelah SMOTE")
    st.write(y_test_smote.value_counts())

# ------------------PreProcessing Data-----------------------
st.markdown(
    '''
    <h1 align="center">PRE PROCESSING DATA</h1>
    '''
    , unsafe_allow_html=True
)
deskPre()
st.subheader('PreProcessing menggunakan SMOTE')
st.code("""
    data = pd.read_excel('wine_quality.xlsx')
 
    X = data.drop('color', axis=1)
    y = data['color']

    # Normalisasi data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # SMOTE untuk mengatasi data tidak seimbang
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    X_test_smote, y_test_smote = smote.fit_resample(X_test, y_test)

    # Tampilkan hasil ke halaman Streamlit
    print("### Data Train Sebelum SMOTE")
    print(y_train.value_counts())
    
    print("### Data Train Setelah SMOTE")
    print(y_train_smote.value_counts())
    
    print("### Data Test Sebelum SMOTE")
    print(y_test.value_counts())
    
    print("### Data Test Setelah SMOTE")
    print(y_test_smote.value_counts())
    """)
pre()
