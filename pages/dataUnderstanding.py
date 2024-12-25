import streamlit as st
import pandas as pd
from component.bootstrap import bootstrap
from component.nav import navbar

# def dataUnderstanding():
# ------------------navbar-----------------------
navbar()

def html():
    bootstrap()
    st.header('Deskripsi Data')
    st.subheader ('Mengetahui Kualitas Anggur (Wine Quality)')
    deskripsi = f"""
    <div class="container">
        <div class="card mt-3">
            <div class="card-body">
                Wine merupakan minuman dengan kandungan alkohol yang terbuat dari fermentasi anaerob sari buah anggur yang didalamnya tidak memiliki kandungan O2. Keseimbangan alami yang dimiliki dalam kandungan buah anggur ini dapat menyebabkan buah anggur tersebut dapat difermentasi tanpa adanya penambahan gula, asam, enzim, atau zat gizi lainnya. Pembuatan wine menggunakan cara memfermentasi sari buah anggur menggunakan ragi tertentu yang kemudian mengandung gula dalam buah anggur tersebut yang akan dikonsumsi oleh ragi dan mengubahnya menjadi alkohol. Salah satu cara yang dapat digunakan untuk memprediksi kualitas dari sebuah wine dapat digunakan dengan cara klasifikasi data menggunakan data mining. Proses klasifikasi data bertujuan untuk melakukan prediksi terhadap kelas dari suatu data yang belum diketahui sebelumnya. Machine Learning ini adalah gabungan dari model faktual yang berkaitan dengan ekstraksi pembelajaran dan model dari dataset yang besar.
            </div>
        </div>
        <br>
        <div class="list-group">
            <p>Dalam dataset ini memiliki beberapa fitur dan 1 class yaitu class Color. <strong>Fitur yang terdapat dalam dataset ini yaitu :</strong></p>
            <ol class="list-group list-group-numbered">
                <li class="list-group-item">fixed_acidity</li>
                <li class="list-group-item">volatile_acidity</li>
                <li class="list-group-item">residual_sugar</li>
                <li class="list-group-item">chlorides</li>
                <li class="list-group-item">free_sulfur_dioxide</li>
                <li class="list-group-item">total_sulfur_dioxide</li>
                <li class="list-group-item">density</li>
                <li class="list-group-item">pH</li>
                <li class="list-group-item">sulphates</li>
                <li class="list-group-item">alcohol</li>
                <li class="list-group-item">quality</li>
                <li class="list-group-item">color</li>
            </ol>
        </div>
    </div>
    """
    st.markdown(deskripsi, unsafe_allow_html=True)
    st.image('asset/wine.png',caption='wine quality', width=500)

def desk_fitur():
    bootstrap()
    fitur = """
    <p>Diatas ini merupakan info dari dataset yang kita gunakan dari nomer 1 hingga 11 merupakan fitur dan quality termasuk kedalam class dari dataset.  Dari info diatas juga menjelaskan <strong>tipe data</strong> yang digunakan yaitu numerik. Berikut penjelasan tentang setiap fitur: </p>

    <div class="list-group">
        <p>Dalam dataset ini memiliki beberapa fitur dan 1 class yaitu class quality. <strong>Fitur yang terdapat dalam dataset ini yaitu :</strong></p>
        <ol class="list-group list-group-numbered">
            <li class="list-group-item"><strong>fixed_acidity</strong> : Jumlah asam tetap yang terkandung didalam wine , dimana kandungan asam tersebut tidak mudah menguap</li>
            <li class="list-group-item"><strong>volatile_acidity</strong> : Jumlah asam asetat mudah menguap yang terkandung dalam wine, dimana pada tingkat konsentrasi terlalu tinggi dapat merusak rasa</li>
            <li class="list-group-item"><strong>residual_sugar</strong> : Jumlah gula yang tersisa setelah fermentasi berhenti</li>
            <li class="list-group-item"><strong>chlorides</strong> : Jumlah garam yang terkandung di dalam wine</li>
            <li class="list-group-item"><strong>free_sulfur_dioxide</strong> : Jumlah kandungan SO2 yang ada dalam wine. Dimana SO2 berbentuk bebas dalam kesetimbangan antara molekul SO2 dan ion bisulfit, yang mencegah pertumbuhan mikroba dan mencegah terjadinya oksidasi pada wine </li>
            <li class="list-group-item"><strong>total_sulfur_dioxide</strong> : Jumlah bentuk bebas dan terikat SO2, dalam konsentrasi rendah, SO2 sebagian besar tidak terdeteksi dalam wine</li>
            <li class="list-group-item"><strong>density</strong> : Kerapatan air yang tergantung pada persen alkohol dan kadar gula dalam wine</li>
            <li class="list-group-item"><strong>pH</strong> : Tingkat asam atau basa wine, kebanyakan wine berada di skala 3-4 pada skala pH</li>
            <li class="list-group-item"><strong>sulphates</strong> : Kadar aditif wine yang dapat berkontribusi pada kadar gas SO2, yang bertindak sebagai anti mikrona dan antioksidan</li>
            <li class="list-group-item"><strong>alcohol</strong> : Persen kandungan alkohol dalam wine</li>
            <li class="list-group-item"><strong>quality</strong> : Variabel output (berdasarkan data sensorik, nilainya antara 0-10)</li>
        </ol>
    </div>
    """
    st.markdown(fitur, unsafe_allow_html=True)

def desk_kelas():
    bootstrap()
    kelas = """
     <div class="list-group">
            <p>Dari hasil diatas menunjukkan bahwa class dalam dataset yang digunakan ada di dalam kolom quality dengan jumlah 6 class, yaitu :</p>
            <ol class="list-group list-group-numbered">
                <li class="list-group-item">3</li>
                <li class="list-group-item">4</li>
                <li class="list-group-item">5</li>
                <li class="list-group-item">6</li>
                <li class="list-group-item">7</li>
                <li class="list-group-item">8</li>
            </ol>
        </div>
    """
    st.markdown(kelas, unsafe_allow_html=True)
    
    
def missval():
    misval="""
    <p>
    Disini kita dapat mengetahui apakah data set yang kita gunakan memiliki missing value atau tidak.
    </p>
    """
    st.markdown(misval, unsafe_allow_html=True)

def penjelasan_missval():
    penjelasan="""
    <p>
    dari hasil diatas kita dapat mengetahui bahwa dataset yang kita gunakan tidak memiliki missing value.
    </p>
    """
    st.markdown(penjelasan, unsafe_allow_html=True)
    
def display():
    st.markdown(
        '<h1 align="center">DATA UNDERSTANDING</h1>'
        ,unsafe_allow_html=True
    )
    html()
    # ------------------FULL DATA-----------------------
    st.subheader ('Menampilkan Data Wine')
    df = pd.read_excel('wine_quality.xlsx')
    st.write(df)

    # ------------------PENJELASAN FITUR-----------------------
    st.subheader('Feature')
    'Jenis data dari setiap kolom:'
    dtypes = pd.DataFrame(df.dtypes, columns=["Tipe Data"])
    st.dataframe(dtypes)

    desk_fitur()

    # ------------------PENJELASAN CLASS-----------------------
    st.subheader('Penjelasan Class')
    df_class = df.value_counts('quality')
    st.write(df_class)

    desk_kelas()

    st.subheader('Cek Missing Value')
    missval()
    df = df.isnull().sum()
    df
    penjelasan_missval()
display()
