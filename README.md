# PANDAWA - Eye Disease Detection 👁️

PANDAWA (**Pendeteksi Awal Penyakit Mata**) adalah aplikasi berbasis web untuk mendeteksi kondisi kesehatan mata secara dini menggunakan teknologi machine learning. Aplikasi ini dapat mengklasifikasikan gambar retina menjadi empat kategori utama: 

- Katarak
- Retinopati Diabetik
- Glaukoma
- Normal

Aplikasi ini menggunakan **EfficientNet** sebagai model dasar dan dilengkapi dengan fitur visualisasi seperti **Saliency Map** untuk membantu memahami hasil prediksi.

## Fitur Utama

1. **Prediksi Kondisi Mata**: 
   - Menggunakan model machine learning untuk mendeteksi kondisi kesehatan mata dari gambar retina.
2. **Visualisasi Probabilitas**:
   - Menampilkan probabilitas prediksi untuk setiap kondisi mata dalam bentuk grafik.
3. **Saliency Map**:
   - Menunjukkan area pada gambar retina yang paling berkontribusi terhadap prediksi model.
4. **Penjelasan Medis**:
   - Memberikan wawasan berbasis medis untuk setiap kondisi mata.

## Teknologi yang Digunakan

- **Streamlit**: Untuk membuat antarmuka web interaktif.
- **PyTorch**: Untuk model machine learning berbasis EfficientNet.
- **Transformers**: Library dari Hugging Face untuk model image classification.
- **Matplotlib**: Untuk visualisasi data.
- **Pillow (PIL)**: Untuk pemrosesan gambar.
- **Torchvision**: Untuk transformasi gambar.

## Instalasi dan Penggunaan


### Langkah Instalasi

1. **Clone repositori ini**:
   ```bash
   git clone https://github.com/JonathanAdriell/tsdn-app.git
   cd tsdn-app
   ```

2. **Buat environment virtual**:
   ```bash
   python -m venv env
   ```

3. **Aktifkan environment virtual**:
   ```bash
   source venv/bin/activate    # Untuk Linux/Mac
   .\venv\Scripts\activate     # Untuk Windows
   ```

4. **Pasang dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Unduh model yang sudah dilatih**:
   - Simpan file model yang dilatih (`efficientnet_model.pth`) di folder `model/`.

6. **Jalankan aplikasi**:
   ```bash
   streamlit run app.py
   ```

7. **Akses aplikasi**:
   - Buka browser dan akses: `http://localhost:8501`


### Aplikasi yang Dideploy

Anda bisa langsung mencoba aplikasi **PANDAWA** yang sudah dideploy secara online. Cukup klik link berikut untuk mengakses informasi dan memanfaatkan fitur prediksi kondisi mata:

[**Coba Aplikasi PANDAWA**](https://eye-disease-classification.streamlit.app/)


### Struktur Proyek

```plaintext
├── app.py                        # Aplikasi Streamlit utama
├── model/
│   └── efficientnet_model.pth    # Model yang sudah dilatih
├── assets/
│   ├── katarak.png               # Gambar untuk deskripsi Katarak
│   ├── retinopati_diabetik.jpeg  # Gambar untuk deskripsi Retinopati Diabetik
│   └── glaukoma.png              # Gambar untuk deskripsi Glaukoma
├── requirements.txt              # Daftar dependensi Python
└── README.md                     # Dokumentasi proyek
```

## Penjelasan Fitur

### 1. **Prediksi Kondisi Mata**
Fitur utama aplikasi ini memungkinkan pengguna untuk mengunggah gambar retina mereka dalam format **JPG**, **PNG**, atau **JPEG**. Setelah diunggah, model
machine learning akan menganalisis gambar tersebut untuk mendeteksi salah satu dari empat kondisi mata.

### 2. **Visualisasi Probabilitas**
Hasil prediksi disertai dengan probabilitas untuk masing-masing kategori yang ditampilkan dalam bentuk **grafik batang horizontal berwarna**. Fitur ini memungkinkan pengguna untuk:

- Memahami tingkat keyakinan model terhadap prediksi yang diberikan.
- Membandingkan kemungkinan untuk setiap kondisi mata berdasarkan probabilitas yang dihitung.

### 3. **Saliency Map**
Saliency map adalah fitur visualisasi yang menunjukkan area gambar retina yang paling berkontribusi pada keputusan model dalam membuat prediksi. Beberapa manfaat dari fitur ini meliputi:

- Membantu pengguna memahami alasan di balik keputusan model.
- Memberikan wawasan tambahan bagi pengguna untuk memverifikasi hasil prediksi.
- Menyoroti area penting seperti:
  - Katarak: Area buram pada retina.
  - Retinopati Diabetik: Pembuluh darah yang rusak atau abnormal.
  - Glaukoma: Perubahan pada cakram optik.
  - Normal: Tidak ada fokus signifikan, menunjukkan retina yang sehat.

### 4. **Penjelasan Medis**
Aplikasi ini dilengkapi dengan penjelasan berbasis medis yang mudah dipahami untuk setiap kondisi mata. Penjelasan ini mencakup:

- Penyebab utama dari kondisi tersebut.
- Gejala klinis yang biasanya muncul pada pasien.
- Pentingnya deteksi dini untuk mencegah komplikasi lebih lanjut.
- Wilayah retina yang relevan dengan kondisi tersebut, seperti cakram optik untuk glaukoma

## Catatan Teknis

- **Model**: Aplikasi menggunakan **EfficientNet-B7** yang dilakukan finetuning untuk klasifikasi empat kondisi mata.
- **Saliency Map**: Menggunakan gradien dari lapisan input gambar untuk menghasilkan peta yang menunjukkan kontribusi masing-masing piksel terhadap prediksi model.

## Penutup

Aplikasi **PANDAWA – Eye Disease Detection** dirancang untuk semua pengguna, baik individu dengan foto retina maupun dokter spesialis.

- **Pengguna Awam**: Mudah digunakan dan membantu mendeteksi dini penyakit mata dengan foto retina untuk konsultasi lebih lanjut.
- **Dokter**: Membantu verifikasi hasil prediksi melalui saliency map dan probabilitas yang diberikan, mendukung diagnosis lebih cepat dan akurat.

Dengan aplikasi ini, teknologi dan dunia medis berkolaborasi untuk meningkatkan akses dan kualitas perawatan kesehatan mata, mencegah kebutaan, dan menyelamatkan penglihatan.

> "Mata adalah jendela dunia. Deteksi dini adalah kunci untuk menjaga keindahannya."  

Selamat menggunakan aplikasi **PANDAWA**! 🌟👁️  
