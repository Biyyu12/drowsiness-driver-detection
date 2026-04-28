# Drowsiness Driver Detection 🚗💤

Drowsiness Driver Detection adalah sebuah sistem berbasis *Computer Vision* yang dirancang untuk mendeteksi tingkat kantuk pengemudi secara *real-time*. Program ini menggunakan kamera (webcam) untuk memantau wajah pengemudi dan akan memberikan peringatan berupa suara alarm apabila mendeteksi tanda-tanda bahaya seperti mata terpejam (*microsleep*) atau menguap, sehingga dapat membantu mencegah terjadinya kecelakaan akibat kelelahan.

## 📌 1. Overview Project

Keselamatan berkendara adalah prioritas utama. Proyek ini dibangun menggunakan Python dan memanfaatkan pemrosesan citra untuk mendeteksi titik-titik (*landmarks*) pada wajah pengemudi. 
Dengan menggunakan antarmuka dari **Streamlit** untuk tampilan web interaktif, aplikasi ini memproses aliran video secara langsung, menganalisis kondisi mata pengemudi, dan memicu peringatan dari folder `sound` jika pengemudi terindikasi tertidur.

## ⚙️ 2. Cara Kerja dan Flow Program

Berikut adalah alur bagaimana program ini mendeteksi kantuk:

1. **Input Video (Webcam)** <br> Sistem mengambil input berupa video real-time dari kamera (webcam).
2. **Face Detection & Landmark Extraction** <br> Setiap *frame* dari video akan diproses oleh model pendeteksi wajah. Model ini akan memetakan *facial landmarks* (titik-titik koordinat pada wajah), dengan fokus utama pada area mata dan mulut.
3. **Kalkulasi EAR (Eye Aspect Ratio)** <br> Program menghitung jarak antara kelopak mata bagian atas dan bawah. 
   * Jika mata terbuka, nilai EAR akan normal.
   * Jika mata mulai tertutup, nilai EAR akan turun secara signifikan.
4. **Prediksi Model** <br> Model deep learning akan mengklasifikasikan kondisi menguap atau normal.
4. **Validasi dan Thresholding** <br>
    * Jika nilai EAR berada di bawah batas wajar (0.20) selama beberapa *frame* berturut-turut, program mengonfirmasi bahwa pengemudi sedang memejamkan mata (mengantuk atau *microsleep*).
    * Jika probabilitas prediksi model menguap lebih dari 50%, program mengonfirmasi bahwa pengemudi sedang menguap.
5. **Trigger Alarm (Peringatan Suara)** Sistem akan secara otomatis membunyikan alarm peringatan melalui modul yang ada untuk membangunkan pengemudi agar kembali waspada.

## Flow Diagram (Simplified)
```
Webcam Input
      ↓
Face Detection (MediaPipe)
      ↓
Landmark Extraction
      ↓
Preprocessing
      ↓
Kalkulasi EAR & Model Prediction
      ↓
Drowsy / Not Drowsy
      ↓
Alarm Trigger (if drowsy)
```
## 🚀 3. Cara Clone dan Menjalankan Program

Pastikan komputer Anda sudah terinstal **Python** (serta **Git**). Karena project ini mendukung manajemen dependensi dengan `uv`, Anda bisa menggunakannya atau tetap memakai `pip` standar.

### Langkah-langkah:

**1. Clone Repository** Buka terminal atau *command prompt*, lalu jalankan perintah berikut:
```bash
git clone https://github.com/Biyyu12/drowsiness-driver-detection.git
cd drowsiness-driver-detection
```

**2. Buat Virtual Environment**
```bash
python -m venv venv
```
Aktivasi:
Windows
```bash
venv\Scripts\activate
```
Linux / Mac
```bash
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Jalankan Program**
```bash
python app.py
```
---

**⚠️ Catatan Penting**
1. Pastikan webcam aktif
2. Gunakan Python versi kompatibel (disarankan 3.10)
3. Hindari konflik dependency seperti:
    * protobuf
    * mediapipe
    * tensorflow
