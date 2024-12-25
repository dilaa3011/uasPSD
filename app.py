import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import streamlit as st
from component.nav import navbar

st.set_page_config(page_title="UAS Proyek Data Sains", layout="wide")
navbar()

# Daftar fitur
fitur = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 
         'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

# Tab untuk prediksi
tabs = st.tabs(["Predict"])

with tabs[0]:
    st.header('Perbandingan metode SVM dan Naive Bayes dalam memprediksi wine')
    st.text('Silahkan Masukkan Data')

    # Inisialisasi list di session state jika belum ada
    if 'data_input' not in st.session_state:
        st.session_state.data_input = [0.0] * len(fitur)
    if 'data_current_index' not in st.session_state:
        st.session_state.data_current_index = 0

    # Form input data
    if st.session_state.data_current_index < len(fitur):
        with st.form(key='data_input_form'):
            feature = fitur[st.session_state.data_current_index]
            angka = st.number_input(f'Masukkan {feature}: ', key=f'{feature}', 
                    value=st.session_state.data_input[st.session_state.data_current_index])
            submit_button = st.form_submit_button(label='Tambah ke Data')
            
            if submit_button:
                st.session_state.data_input[st.session_state.data_current_index] = angka
                st.session_state.data_current_index += 1
                st.experimental_rerun()

    # Menampilkan data input
    st.write('Data:', st.session_state.data_input)

    # Jika semua fitur sudah diinput
    if st.session_state.data_current_index == len(fitur):
        df = pd.read_excel('wine_quality.xlsx')

        # Memisahkan fitur dan target
        X = df[fitur]
        y = df['color']

        # Normalisasi
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

        # SMOTE 
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # Inisialisasi model
        models = {
            "SVM (Linear Kernel)": SVC(kernel='linear', random_state=42),
            "SVM (RBF Kernel)": SVC(kernel='rbf', random_state=42),
            "SVM (Polynomial Kernel)": SVC(kernel='poly', degree=3, random_state=42),
            "Naive Bayes": GaussianNB()
        }

        # hasil evaluasi
        results = []

        for name, model in models.items():
            model.fit(X_train_smote, y_train_smote)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            accuracy = accuracy_score(y_test, y_pred)

            # menambah hasil ke list
            results.append({
                "Model": name,
                "Precision": report['weighted avg']['precision'],
                "Recall": report['weighted avg']['recall'],
                "F1-Score": report['weighted avg']['f1-score'],
                "Accuracy (%)": accuracy * 100
            })

        results_df = pd.DataFrame(results)
        st.write("## Hasil Perbandingan Model")
        st.table(results_df)

        # Prediksi untuk data baru
        X_new = scaler.transform([st.session_state.data_input])
        predictions = {name: model.predict(X_new)[0] for name, model in models.items()}
        
        st.write("## Prediksi untuk Data Baru")
        prediction_table = pd.DataFrame({
            "Model": list(predictions.keys()),
            "Predicted Class": list(predictions.values())
        })
        st.table(prediction_table)

