import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

from component.nav import navbar

    # ------------------navbar-----------------------
navbar()
def svm():
    st.header('Modelling Support Vector Machine (SVM)')
    desk_svm = """
    <p>
    Metode SVM digunakan untuk mengukur perbedaan antara fitur normal dan abnormal, hal ini didukung oleh kemampuan SVM yang dapat menangani data nonlinier[6]. Konsep inti SVM dalam klasifikasi melibatkan identifikasi pemisah optimal, yang dikenal sebagai hyperplane, antara dua kelas data. Hyperplane dianggap optimal jika menawarkan margin terbesar, yang mewakili dua kali jarak antara hyperplane dan support vector.
    </p>
    """
    st.markdown(desk_svm,unsafe_allow_html=True)
    
    # Mengakses dataset melalui csv
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

    
    st.subheader('Menghitung Akurasi Model')
    # Melatih model SVM dengan kernel Linear
    svm_linear = SVC(kernel='linear',C=100, probability=True, random_state=42)
    svm_linear.fit(X_train_smote, y_train_smote)

    # Prediksi pada data uji
    y_pred_linear = svm_linear.predict(X_test)

    y_proba_linear = svm_linear.predict_proba(X_test)[:, 1]  # Probabilitas untuk kelas positif

    # Model dengan SVM menggunakan kernel Polynomial
    svm_poly = SVC(kernel='poly', degree=3, random_state=42)  # Degree bisa disesuaikan
    svm_poly.fit(X_train_smote, y_train_smote)

    # Evaluasi model dengan kernel Polynomial
    y_pred_poly = svm_poly.predict(X_test)
    print("\nPolynomial Kernel:")
    # print(confusion_matrix(y_test, y_pred_poly))
    print(classification_report(y_test, y_pred_poly))

    # Model dengan SVM menggunakan kernel RBF (untuk perbandingan)
    svm_rbf = SVC(kernel='rbf', random_state=42)
    svm_rbf.fit(X_train_smote, y_train_smote)

    # Evaluasi model dengan kernel RBF
    y_pred_rbf = svm_rbf.predict(X_test)

    print("\nRBF Kernel:")
    # print(confusion_matrix(y_test, y_pred_rbf))
    print(classification_report(y_test, y_pred_rbf))

    # Model Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train_smote, y_train_smote)

    # Prediksi dan evaluasi model
    y_pred = nb_model.predict(X_test)

    # Print classification report
    accuracy_nb = accuracy_score(y_test, y_pred)
    accuracy_svm_rbf = accuracy_score(y_test, y_pred_rbf)
    accuracy_svm_poly = accuracy_score(y_test, y_pred_poly)
    accuracy_svm_linear = accuracy_score(y_test, y_pred_linear)

    # Menyusun data akurasi dalam bentuk dictionary
    accuracy_data = {
        'Model': ['Naive Bayes', 'SVM RBF', 'SVM Polynomial', 'SVM Linear'],
        'Accuracy': [accuracy_nb, accuracy_svm_rbf, accuracy_svm_poly, accuracy_svm_linear]
    }

    # Membuat DataFrame
    accuracy_df = pd.DataFrame(accuracy_data)

    # Menampilkan tabel akurasi
    st.write(accuracy_df)
    # st.write('Model svm dengan kernel Linier memiliki akurasi sebesar : ', accuracy_svm_linear)

    st.subheader('Menghitung akurasi menggunakan model svm dengan kernel Linier')
    st.code("""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score


    # Melatih model SVM dengan kernel Linear
    svm_linear = SVC(kernel='linear',C=100, probability=True, random_state=42)
    svm_linear.fit(X_train_smote, y_train_smote)

    # Prediksi pada data uji
    y_pred_linear = svm_linear.predict(X_test)

    y_proba_linear = svm_linear.predict_proba(X_test)[:, 1]  # Probabilitas untuk kelas positif

    # Print classification report
    print("SVM for Linear Kernel:")
    print(classification_report(y_test, y_pred_linear))

            """)
    
    st.subheader('Menghitung akurasi menggunakan model svm dengan kernel Polynomial')
    st.code("""
    # Model dengan SVM menggunakan kernel Polynomial
    svm_poly = SVC(kernel='poly', degree=3, random_state=42)  # Degree bisa disesuaikan
    svm_poly.fit(X_train_smote, y_train_smote)

    # Evaluasi model dengan kernel Polynomial
    y_pred_poly = svm_poly.predict(X_test)
    print("\nPolynomial Kernel:")
    # print(confusion_matrix(y_test, y_pred_poly))
    print(classification_report(y_test, y_pred_poly))
            """)
    
    st.subheader('Menghitung akurasi menggunakan model svm dengan kernel RBF')
    st.code("""
    # Model dengan SVM menggunakan kernel RBF (untuk perbandingan)
    svm_rbf = SVC(kernel='rbf', random_state=42)
    svm_rbf.fit(X_train_smote, y_train_smote)

    # Evaluasi model dengan kernel RBF
    y_pred_rbf = svm_rbf.predict(X_test)

    print("\nRBF Kernel:")
    # print(confusion_matrix(y_test, y_pred_rbf))
    print(classification_report(y_test, y_pred_rbf))
            """)

# ----------------------------------------------------------------------------------------------------------------------------------------------
def bayes():
    st.header('GAUSSIAN NAIVE BAYES CLASSIFICATION')
    'Naïve Bayes merupakan algoritma klasifikasi pembelajaran mesin yang umum digunakan. Pengklasifikasi Naïve Bayes merupakan pengklasifikasi probabilistik sederhana dengan menerapkan teorema Bayes dengan mengabaikan ketergantungan antar fitur[5]. Pada algoritma Naive Bayes saat klasifikasi tidak membutuhkan adanya pemodelan maupun uji statistik. Naive Bayes salah satu metode machine learning yang menggunakan perhitungan probabilitas.'
    'Sebelum memulai menghitung tentunya kita perlu menyiapkan data yang akan kita pakai.'

    df = pd.read_excel('wine_quality.xlsx')
    st.write(df.head())

    'Untuk mempermudah perhitungan kita akan membagi data menjadi 20% data sebagai data test dan 80% data sebagai data train. Hal ini dilakukan agar saat kita melakukan prediksi terhadap data baru, kita mendapat hasil yang lebih efektif.'
    st.subheader('Menghitung Akurasi Menggunakan Naive Bayes')
    st.code("""
    from sklearn.naive_bayes import GaussianNB
            
    # Model Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train_smote, y_train_smote)

    # Prediksi dan evaluasi model
    y_pred = nb_model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    """)
# ----------------------------------------------------------------------------------------------------------------------------------------------
def model():
    # ------------------Modelling-----------------------
    st.markdown(
        '<h1 align="center">Modelling</h1>'
        ,unsafe_allow_html=True
    )
    svm()
    bayes()


    
model()