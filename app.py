# import library
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# import file csv
url = ("https://raw.githubusercontent.com/Ahmad08017928/app_dataset_smk.github.io/main/dataset/data.csv")
df = pd.read_csv(url)

# data cleaning
df1 = df.drop(['Timestamp', '14. Orangtua saya menunjukkan bagaimana menggunakan kelebihan saya pada berbagai situasi yang berbeda.'], axis='columns')
#Fill in the empty data with the previous value
data = df1.copy()
data[['NAMA LENGKAP', 'JENIS KELAMIN', 'KELAS',
'INFORMASI TENTANG ORANG TUA ANDA',
'APAKAH SAAT INI ANDA YATIM PIATU?',
'SAAT INI ANDA DIASUH OLEH SIAPA? ',
'1. Orangtua saya memahami kelebihan-kelebihan saya (kepribadian',
' kemampuan', ' bakat dan keterampilan).',
'2. Orangtua saya tidak mengetahui kelebihan-kelebihan saya.',
'3. Orangtua saya mengetahui kemampuan terbaik saya.',
'4. Orangtua saya menyadari kelebihan-kelebihan saya.',
'5. Orang tua saya mengetahui hal-hal yang bisa saya lakukan dengan baik.',
'6. Orangtua saya mengetahui dengan baik kelebihan-kelebihan saya.',
'7. Orangtua saya memahami kemampuan terbaik saya.',
'8. Orangtua saya memberi kesempatan untuk secara rutin melakukan hal terbaik yang bisa saya lakukan.',
'9. Orangtua mendorong saya untuk selalu menggunakan kelebihan-kelebihan saya.',
'10. Orangtua saya mendorong untuk melakukan hal yang saya kuasai dengan baik.',
'11. Orangtua saya menyarankan agar saya memanfaatkan kelebihan-kelebihan saya.',
'12. Orangtua saya memberi banyak kesempatan untuk menggunakan kelebihan-kelebihan saya.',
'13. Orangtua saya membantu saya memikirkan cara menggunakan kelebihan-kelebihan saya.']] = data[['NAMA LENGKAP', 'JENIS KELAMIN', 'KELAS',
'INFORMASI TENTANG ORANG TUA ANDA',
'APAKAH SAAT INI ANDA YATIM PIATU?',
'SAAT INI ANDA DIASUH OLEH SIAPA? ',
'1. Orangtua saya memahami kelebihan-kelebihan saya (kepribadian',
' kemampuan', ' bakat dan keterampilan).',
'2. Orangtua saya tidak mengetahui kelebihan-kelebihan saya.',
'3. Orangtua saya mengetahui kemampuan terbaik saya.',
'4. Orangtua saya menyadari kelebihan-kelebihan saya.',
'5. Orang tua saya mengetahui hal-hal yang bisa saya lakukan dengan baik.',
'6. Orangtua saya mengetahui dengan baik kelebihan-kelebihan saya.',
'7. Orangtua saya memahami kemampuan terbaik saya.',
'8. Orangtua saya memberi kesempatan untuk secara rutin melakukan hal terbaik yang bisa saya lakukan.',
'9. Orangtua mendorong saya untuk selalu menggunakan kelebihan-kelebihan saya.',
'10. Orangtua saya mendorong untuk melakukan hal yang saya kuasai dengan baik.',
'11. Orangtua saya menyarankan agar saya memanfaatkan kelebihan-kelebihan saya.',
'12. Orangtua saya memberi banyak kesempatan untuk menggunakan kelebihan-kelebihan saya.',
'13. Orangtua saya membantu saya memikirkan cara menggunakan kelebihan-kelebihan saya.']].fillna(method='pad')

 # Trasformasi Data
Numerik = LabelEncoder()
x = data.drop(['NAMA LENGKAP', 'SAAT INI ANDA DIASUH OLEH SIAPA? ', '1. Orangtua saya memahami kelebihan-kelebihan saya (kepribadian',' kemampuan'], axis='columns')
y = data[[' kemampuan']]

x['JENIS KELAMIN_n'] = Numerik.fit_transform(x['JENIS KELAMIN'])
x['INFORMASI TENTANG ORANG TUA ANDA_n'] = Numerik.fit_transform(x['INFORMASI TENTANG ORANG TUA ANDA'])
x['APAKAH SAAT INI ANDA YATIM PIATU?_n'] = Numerik.fit_transform(x['APAKAH SAAT INI ANDA YATIM PIATU?'])
x['bakat dan keterampilan_n'] = Numerik.fit_transform(x[' bakat dan keterampilan).'])
x['2. Orangtua saya tidak mengetahui kelebihan-kelebihan saya._n'] = Numerik.fit_transform(x['2. Orangtua saya tidak mengetahui kelebihan-kelebihan saya.'])
x['3. Orangtua saya mengetahui kemampuan terbaik saya._n'] = Numerik.fit_transform(x['3. Orangtua saya mengetahui kemampuan terbaik saya.'])
x['4. Orangtua saya menyadari kelebihan-kelebihan saya._n'] = Numerik.fit_transform(x['4. Orangtua saya menyadari kelebihan-kelebihan saya.'])
x['5. Orang tua saya mengetahui hal-hal yang bisa saya lakukan dengan baik._n'] = Numerik.fit_transform(x['5. Orang tua saya mengetahui hal-hal yang bisa saya lakukan dengan baik.'])
x['6. Orangtua saya mengetahui dengan baik kelebihan-kelebihan saya._n'] = Numerik.fit_transform(x['6. Orangtua saya mengetahui dengan baik kelebihan-kelebihan saya.'])
x['7. Orangtua saya memahami kemampuan terbaik saya._n'] = Numerik.fit_transform(x['7. Orangtua saya memahami kemampuan terbaik saya.'])
x['8. Orangtua saya memberi kesempatan untuk secara rutin melakukan hal terbaik yang bisa saya lakukan._n'] = Numerik.fit_transform(x['8. Orangtua saya memberi kesempatan untuk secara rutin melakukan hal terbaik yang bisa saya lakukan.'])
x['9. Orangtua mendorong saya untuk selalu menggunakan kelebihan-kelebihan saya._n'] = Numerik.fit_transform(x['9. Orangtua mendorong saya untuk selalu menggunakan kelebihan-kelebihan saya.'])
x['10. Orangtua saya mendorong untuk melakukan hal yang saya kuasai dengan baik._n'] = Numerik.fit_transform(x['10. Orangtua saya mendorong untuk melakukan hal yang saya kuasai dengan baik.'])
x['11. Orangtua saya menyarankan agar saya memanfaatkan kelebihan-kelebihan saya._n'] = Numerik.fit_transform(x['11. Orangtua saya menyarankan agar saya memanfaatkan kelebihan-kelebihan saya.'])
x['12. Orangtua saya memberi banyak kesempatan untuk menggunakan kelebihan-kelebihan saya._n'] = Numerik.fit_transform(x['12. Orangtua saya memberi banyak kesempatan untuk menggunakan kelebihan-kelebihan saya.'])
x['13. Orangtua saya membantu saya memikirkan cara menggunakan kelebihan-kelebihan saya._n'] = Numerik.fit_transform(x['13. Orangtua saya membantu saya memikirkan cara menggunakan kelebihan-kelebihan saya.'])
y['kemampuan_n'] = Numerik.fit_transform(y[' kemampuan'])

x_d = x.drop(['JENIS KELAMIN',
'INFORMASI TENTANG ORANG TUA ANDA', 'APAKAH SAAT INI ANDA YATIM PIATU?', ' bakat dan keterampilan).',
'2. Orangtua saya tidak mengetahui kelebihan-kelebihan saya.',
'3. Orangtua saya mengetahui kemampuan terbaik saya.',
'4. Orangtua saya menyadari kelebihan-kelebihan saya.',
'5. Orang tua saya mengetahui hal-hal yang bisa saya lakukan dengan baik.',
'6. Orangtua saya mengetahui dengan baik kelebihan-kelebihan saya.',
'7. Orangtua saya memahami kemampuan terbaik saya.',
'8. Orangtua saya memberi kesempatan untuk secara rutin melakukan hal terbaik yang bisa saya lakukan.',
'9. Orangtua mendorong saya untuk selalu menggunakan kelebihan-kelebihan saya.',
'10. Orangtua saya mendorong untuk melakukan hal yang saya kuasai dengan baik.',
'11. Orangtua saya menyarankan agar saya memanfaatkan kelebihan-kelebihan saya.',
'12. Orangtua saya memberi banyak kesempatan untuk menggunakan kelebihan-kelebihan saya.',
'13. Orangtua saya membantu saya memikirkan cara menggunakan kelebihan-kelebihan saya.'], axis='columns')
y_d = y.drop(" kemampuan", axis='columns')

# training 75 persen dan testing 25 persen
x_train, x_test, y_train, y_test = train_test_split(x_d, y_d, test_size=0.25, random_state=500)

# Classification knn
classification = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classification.fit(x_train, y_train)

#classification svm
clf = SVC(kernel='linear', probability=True)
clf.fit(x_train, y_train)

#classification Random Forest
Class_RandomFores = RandomForestClassifier()
Class_RandomFores.fit(x_train, y_train)

Nama = st.text_input("Nama")
jk   = st.radio('jenis Kelamin', options=("laki laki", "perempuan"))
kelas= st.radio('Kelas', options=("X", "XI", "XII"))
status_orang_tua = st.radio('Status orang tua', options =('ORANG TUA LENGKAP', 'CERAI HIDUP', 'CERAI MATI'))
status_yatim_piatu = st.radio('Status yatim piatu', options=('TIDAK', 'PIATU', 'YATIM', 'YATIM PIATU'))
Status_1 = st.radio('Apakah Orang tua anda mengetahui bakat dan keterampilan anda ?', options=('Sesuai', 'Sangat Tidak Sesuai', 'Tidak Sesuai', 'Sangat Sesuai'))
Status_2 = st.radio('Apakah Orang tua anda tidak mengetahui kelebihan kelebihan anda ?', options=('Sesuai', 'Tidak Sesuai', 'Sangat Sesuai', 'Sangat Tidak Sesuai'))
Status_3 = st.radio('Apakah Orang tua anda mengetahui kemampuan terbaik anda ? ', options=('Sesuai', 'Tidak Sesuai', 'Sangat Sesuai', 'Sangat Tidak Sesuai'))
Status_4 = st.radio('Apakah Orang tua anda menyadari kelebihan kelebihan anda ?', options=('Sesuai', 'Sangat Sesuai', 'Tidak Sesuai', 'Sangat Tidak Sesuai'))
Status_5 = st.radio('Apakah Orang tua anda mengetahui hal yang bisa anda lakukan ?', options=('Sesuai', 'Tidak Sesuai', 'Sangat Sesuai', 'Sangat Tidak Sesuai'))
Status_6 = st.radio('Apakah Orang tua anda mengetahui dengan baik kelebihan anda ?', options=('Sesuai', 'Tidak Sesuai', 'Sangat Sesuai', 'Sangat Tidak Sesuai'))
Status_7 = st.radio('Apakah Orang tua anda memahami kemampuan terbaik anda ?', options=('Sesuai', 'Sangat Sesuai', 'Sangat Tidak Sesuai', 'Tidak Sesuai'))
Status_8 = st.radio('Apakah orang tua anda memberikan kesempatan pada anda untuk melakukan hal terbaik yang bisa anda lakukan ? ', options=('Sesuai', 'Tidak Sesuai', 'Sangat Sesuai', 'Sangat Tidak Sesuai'))
Status_9 = st.radio('Apakah orang tua anda mendorong anda untuk selalu menggunakan kelebihan anda ?', options=('Sesuai', 'Sangat Sesuai', 'Tidak Sesuai', 'Sangat Tidak Sesuai'))
Status_10 = st.radio('Apakah orang tua anda mendorong anda untuk melakukan hal yang anda kuasai dengan baik ?', options=('Sesuai', 'Sangat Sesuai', 'Tidak Sesuai', 'Sangat Tidak Sesuai'))
Status_11 = st.radio('Apakah orang tua anda menyarankan anda agar memanfaatkan kelebihan kelebihan anda ? ', options=('Sesuai', 'Tidak Sesuai', 'Sangat Sesuai', 'Sangat Tidak Sesuai'))
Status_12 = st.radio('Apakah orang tua anda memberikan anda kesempatan untuk menggunakan kelebihan anda ?', options=('Sesuai', 'Tidak Sesuai', 'Sangat Sesuai', 'Sangat Tidak Sesuai'))
Status_13 = st.radio('Apakah orang tua anda membantu anda memikirkan cara untuk menggunakan kelebihan anda ?', options=('Sesuai', 'Tidak Sesuai', 'Sangat Sesuai', 'Sangat Tidak Sesuai'))

if st.button('Klassifikasi'):
    mybar = st.progress(0)
    jk = 0 if jk == 'laki laki' else 1
    # kelas = 0
    if kelas == 'X':
        kelas = 0
    elif kelas == 'XI':
        kelas == 1
    elif kelas == 'XII':
        kelas = 2

    # status_orang_tua == 0
    if status_orang_tua == 'ORANG TUA LENGKAP':
        status_orang_tua = 2
    elif status_orang_tua == 'CERAI HIDUP':
        status_orang_tua = 0
    elif status_orang_tua == 'CERAI MATI':
        status_orang_tua == 1

    # status_yatim_piatu == 0
    if status_yatim_piatu == 'TIDAK':
        status_yatim_piatu = 1
    elif status_yatim_piatu == 'YATIM':
        status_yatim_piatu = 2
    elif status_yatim_piatu == 'YATIM PIATU':
        status_yatim_piatu = 3
    elif status_yatim_piatu == 'PIATU':
        status_yatim_piatu = 0

    if Status_1 == 'Sesuai':
        Status_1 = 2
    elif Status_1 == 'Sangat Tidak Sesuai':
        Status_1 = 1
    elif Status_1 == 'Tidak Sesuai':
        Status_1 = 3
    elif Status_1 == "Sangat Sesuai":
        Status_1 = 0

    if Status_2 == 'Sesuai':
        Status_2 = 2
    elif Status_2 == 'Tidak Sesuai':
        Status_2 = 3
    elif Status_2 == 'Sangat Tidak Sesuai':
        Status_2 = 1
    elif Status_2 == 'Sangat Sesuai':
        Status_2 = 0

    if Status_3 == 'Sesuai':
        Status_3 = 2
    elif Status_3 == 'Tidak Sesuai':
        Status_3 = 3
    elif Status_3 == 'Sangat Tidak Sesuai':
        Status_3 == 1
    elif Status_3 == 'Sangat Sesuai':
        Status_3 = 0

    if Status_4 == 'Sesuai':
        Status_4 = 2
    elif Status_4 == 'Tidak Sesuai':
        Status_4 = 3
    elif Status_4 == 'Sangat Sesuai':
        Status_4 = 1
    elif Status_4 == 'Sangat Tidak Sesuai':
        Status_4 = 0

    if Status_5 == 'Sesuai':
        Status_5 = 2
    elif Status_5 == 'Tidak Sesuai':
        Status_5 = 3
    elif Status_5 == 'Sangat Tidak Sesuai':
        Status_5 = 1
    elif Status_5 == 'Sangat Sesuai':
        Status_5 = 0

    if Status_6 == 'Sesuai':
        Status_6 = 2
    elif Status_6 == 'Tidak Sesuai':
        Status_6 = 3
    elif Status_6 == 'Sangat Tidak Sesuai':
        Status_6 = 1
    elif Status_6 == 'Sangat Sesuai':
        Status_6 = 0

    if Status_7 == 'Sesuai':
        Status_7 = 2
    elif Status_7 == 'Tidak Sesuai':
        Status_7 = 3
    elif Status_7 == 'Sangat Tidak Sesuai':
        Status_7 = 1
    elif Status_7 == 'Sangat Sesuai':
        Status_7 = 0

    if Status_8 == 'Sesuai':
        Status_8 = 2
    elif Status_8 == 'Tidak Sesuai':
        Status_8 = 3
    elif Status_8 == 'Sangat Tidak Sesuai':
        Status_8 = 1
    elif Status_8 == 'Sangat Sesuai':
        Status_8 = 0

    if Status_9 == 'Sesuai':
        Status_9 = 2
    elif Status_9 == 'Tidak Sesuai':
        Status_9 = 3
    elif Status_9 == 'Sangat Tidak Sesuai':
        Status_9 = 1
    elif Status_9 == 'Sangat Sesuai':
        Status_9 = 0

    if Status_10 == 'Sesuai':
        Status_10 = 2
    elif Status_10 == 'Tidak Sesuai':
        Status_10 = 3
    elif Status_10 == 'Sangat Tidak Sesuai':
        Status_10 = 1
    elif Status_10 == 'Sangat Sesuai':
        Status_10 = 0

    if Status_11 == 'Sesuai':
        Status_11 = 2
    elif Status_11 == 'Tidak Sesuai':
        Status_11 = 3
    elif Status_11 == 'Sangat Tidak Sesuai':
        Status_11 = 1
    elif Status_11 == 'Sangat Sesuai':
        Status_11 = 0

    if Status_12 == 'Sesuai':
        Status_12 = 2
    elif Status_12 == 'Tidak Sesuai':
        Status_12 = 3
    elif Status_12 == 'Sangat Tidak Sesuai':
        Status_12 = 1
    elif Status_12 == 'Sangat Sesuai':
        Status_12 = 0

    if Status_13 == 'Sesuai':
        Status_13 = 2
    elif Status_13 == 'Tidak Sesuai':
        Status_13 = 3
    elif Status_13 == 'Sangat Tidak Sesuai':
        Status_13 = 1
    elif Status_13 == 'Sangat Sesuai':
        Status_13 = 0

    tes = [[jk, kelas, status_orang_tua, status_yatim_piatu, Status_1, Status_2, Status_3, Status_4, Status_5, Status_6, Status_7, Status_8, Status_9, Status_10, Status_11, Status_12, Status_13]]

    hasil_knn = classification.predict(tes)
    akurasi_knn = classification.predict_proba(tes)
    hasil_svm = clf.predict(tes)
    akurasi_svm = clf.predict_proba(tes)
    hasil_RandomForest = Class_RandomFores.predict(tes)
    akurasi_RandomForest = Class_RandomFores.predict_proba(tes)

    import time
    for persen in range (100):
        time.sleep(0.01)
        mybar.progress(persen+1)

        st.subheader("K-Nearest Neighbor")
        if hasil_knn [0]== 2:
            st.write("Orang tua {} mengetahui kemampuan {}, dengan tingkat prediksi {}%".format(Nama, 
            Nama, round(akurasi_knn[0][hasil_knn[0]]*100),3))
        elif hasil_knn == 0:
            st.write("Orang tua {} sangat mengetahui kemampuan {}, dengan tingkat prediksi {}%".format(Nama, Nama, round(akurasi_knn[0][hasil_knn[0]]*100), 3))
        elif hasil_knn == 3 :
            st.write("Orang tua {} tidak mengetahui kemampuan {}, dengan tingkat prediksi {}%".format(Nama, Nama, round(akurasi_knn[0][hasil_knn[0]]*100), 3))
        elif hasil_knn == 1 :
            st.write("Orang tua {} sangat tidak mengetahui kemampuan {}, dengan tingkat prediksi {}%".format(Nama, Nama, round(akurasi_knn[0][hasil_knn[0]]*100), 3))

    st.subheader("Support Vector Machine")
    if hasil_svm [0] == 0:
        st.write("Orang tua {} sangat mengetahui kemampuan {}, dengan tingkat prediksi {}%".format(Nama, Nama, round(akurasi_svm[0][hasil_svm[0]]*100), 3))
    elif hasil_svm == 2:
        st.write("Orang tua {} mengetahui kemampuan {}, dengan tingkat prediksi {}%".format(Nama, Nama, round(akurasi_svm[0][hasil_svm[0]]*100), 3))
    elif hasil_svm == 3 :
        st.write("Orang tua {} tidak mengetahui kemampuan {}, dengan tingkat prediksi {}%".format(Nama, Nama, round(round(akurasi_svm[0][hasil_svm[0]]*100), 3)))
    elif hasil_svm == 1 :
        st.write("Orang tua {} sangat tidak mengetahui kemampuan {}, dengan tingkat prediksi {}%".format(Nama, Nama, round(akurasi_svm[0][hasil_svm[0]]*100), 3))

    st.subheader("Random Forest")
    if hasil_RandomForest [0] == 0:
        st.write("Orang tua {} sangat mengetahui kemampuan {}, dengan tingkat prediksi {}%".format(Nama, Nama, round(akurasi_RandomForest[0][hasil_RandomForest[0]]*100), 3))
    elif hasil_svm == 2:
        st.write("Orang tua {} mengetahui kemampuan {}, dengan tingkat prediksi {}%".format(Nama, Nama, round(akurasi_RandomForest[0][hasil_RandomForest[0]]*100), 3))
    elif hasil_svm == 3 :
        st.write("Orang tua {} tidak mengetahui kemampuan {}, dengan tingkat prediksi {}%".format(Nama, Nama, round(akurasi_RandomForest[0][hasil_RandomForest[0]]*100), 3))
    elif hasil_svm == 1:
        st.write("Orang tua {} sangat tidak mengetahui kemampuan {}, dengan tingkat prediksi {}%".format(Nama, Nama, round(akurasi_RandomForest[0][hasil_RandomForest[0]]*100), 3))
